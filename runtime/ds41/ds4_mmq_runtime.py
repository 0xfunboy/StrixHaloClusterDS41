from __future__ import annotations
import ctypes, os
from dataclasses import dataclass
from typing import Any
import torch

_LIB=None; _FN=None; _INIT=False; _SCRATCH={}
_STATS={'calls':0,'tokens':0,'fallbacks':{},'max_scratch_bytes':0}

def _fallback(reason:str):
    d=_STATS['fallbacks']; d[reason]=int(d.get(reason,0))+1

def stats()->dict[str,Any]:
    return {'calls':int(_STATS['calls']),'tokens':int(_STATS['tokens']),'fallbacks':dict(_STATS['fallbacks']),'max_scratch_bytes':int(_STATS['max_scratch_bytes'])}

def _load():
    global _LIB,_FN,_INIT
    if _FN is not None: return _FN
    path=os.environ.get('DS41_DS4_MMQ_LIB','')
    if not path: raise RuntimeError('DS41_DS4_MMQ_LIB is not set')
    if not os.path.isfile(path): raise RuntimeError(f'DS41 DS4 MMQ library missing: {path}')
    lib=ctypes.CDLL(path,mode=ctypes.RTLD_GLOBAL)
    lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int
    rc=lib.ds4_mmq_init(0)
    if rc: raise RuntimeError(f'ds4_mmq_init rc={rc}')
    fn=lib.ds41_ds4_moe_full_w13
    P=ctypes.c_void_p
    fn.argtypes=[P,P,P,P,P,P, ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float, P,P,P,P,P,P,P,P]
    fn.restype=ctypes.c_int
    _LIB=lib; _FN=fn; _INIT=True
    return fn

def _metadata_counts() -> tuple[int,int] | None:
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id
        if not is_forward_context_available(): return None
        md=get_forward_context().attn_metadata
        if isinstance(md,list): md=md[dbo_current_ubatch_id()]
        vals=list(md.values()) if isinstance(md,dict) else [md]
        pairs=[]
        for v in vals:
            if hasattr(v,'num_decode_tokens') and hasattr(v,'num_prefill_tokens'):
                pairs.append((int(v.num_decode_tokens),int(v.num_prefill_tokens)))
        if not pairs: return None
        first=pairs[0]
        if any(p!=first for p in pairs): return None
        return first
    except Exception:
        return None

def _eligible(layer,x,topk_weights,topk_ids,weight_type,weight_type2,activation):
    if os.environ.get('DS41_DS4_MMQ_PREFILL','0')!='1': return False,'disabled'
    counts=_metadata_counts()
    if counts is None: return False,'no_forward_prefill_metadata'
    nd,np=counts
    if nd!=0 or np!=int(x.shape[0]): return False,'not_pure_target_prefill'
    n=int(x.shape[0]); min_t=int(os.environ.get('DS41_DS4_MMQ_MIN_TOKENS','128')); max_t=int(os.environ.get('DS41_DS4_MMQ_MAX_TOKENS','1024'))
    if n<min_t or n>max_t: return False,'prefill_tokens_outside_admitted_range'
    if x.dtype!=torch.bfloat16 or x.ndim!=2 or x.shape[1]!=5120 or not x.is_contiguous(): return False,'activation_contract'
    if activation!='silu': return False,'activation_not_silu'
    if int(weight_type)!=16 or int(weight_type2)!=10: return False,'quant_contract'
    if topk_ids.ndim!=2 or tuple(topk_ids.shape)!=(n,6) or tuple(topk_weights.shape)!=(n,6): return False,'topk_contract'
    em=getattr(layer,'expert_map',None)
    if em is None or em.dtype!=torch.int32 or em.numel()!=384: return False,'expert_map_contract'
    w13=layer.w13_weight; w2=layer.w2_weight
    if not w13.is_contiguous() or not w2.is_contiguous(): return False,'weight_contiguity'
    if w13.ndim!=3 or w2.ndim!=3 or w13.shape[0]!=w2.shape[0] or w13.shape[1]!=4608 or w2.shape[1]!=5120: return False,'weight_shape'
    return True,'eligible'

@dataclass
class Scratch:
    cap:int; hidden:int; mid:int; device:torch.device
    local_ids:torch.Tensor; x_f32:torch.Tensor; gate:torch.Tensor; up:torch.Tensor; midbuf:torch.Tensor; routes:torch.Tensor; out:torch.Tensor

def _scratch(device:torch.device,cap:int=1024,hidden:int=5120,mid:int=2304)->Scratch:
    key=(device.index,cap,hidden,mid)
    s=_SCRATCH.get(key)
    if s is not None: return s
    local_ids=torch.empty((cap,6),device=device,dtype=torch.int32)
    x_f32=torch.empty((cap,hidden),device=device,dtype=torch.float32)
    gate=torch.empty((cap*6,mid),device=device,dtype=torch.float32); up=torch.empty_like(gate); midbuf=torch.empty_like(gate)
    routes=torch.empty((cap*6,hidden),device=device,dtype=torch.float32); out=torch.empty((cap,hidden),device=device,dtype=torch.float32)
    s=Scratch(cap,hidden,mid,device,local_ids,x_f32,gate,up,midbuf,routes,out); _SCRATCH[key]=s
    total=sum(t.numel()*t.element_size() for t in (local_ids,x_f32,gate,up,midbuf,routes,out)); _STATS['max_scratch_bytes']=max(int(_STATS['max_scratch_bytes']),int(total))
    return s

def _ptr(t:torch.Tensor): return ctypes.c_void_p(t.data_ptr())

def try_apply(method_self,layer,x,topk_weights,topk_ids,weight_type,weight_type2,activation,swiglu_limit):
    ok,reason=_eligible(layer,x,topk_weights,topk_ids,weight_type,weight_type2,activation)
    if not ok:
        _fallback(reason); return None,reason
    fn=_load(); n=int(x.shape[0]); s=_scratch(x.device,cap=int(os.environ.get('DS41_DS4_MMQ_MAX_TOKENS','1024')))
    stream=ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
    rc=fn(_ptr(x),_ptr(topk_ids),_ptr(topk_weights),_ptr(layer.expert_map),_ptr(layer.w13_weight),_ptr(layer.w2_weight),
          n,int(x.shape[1]),2304,384,int(layer.w13_weight.shape[0]),ctypes.c_float(float(swiglu_limit)),
          _ptr(s.local_ids),_ptr(s.x_f32),_ptr(s.gate),_ptr(s.up),_ptr(s.midbuf),_ptr(s.routes),_ptr(s.out),stream)
    if rc:
        raise RuntimeError(f'DS41 DS4 MMQ eligible path failed rc={rc} tokens={n}')
    out=s.out[:n].to(torch.bfloat16)
    _STATS['calls']+=1; _STATS['tokens']+=n
    return out,'ds4_mmq'
