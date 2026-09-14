#!/usr/bin/env python3
"""Real DSpark weight gate on gfx1151: MXFP8 linear + exact MXFP4->BF16 expert fallback."""
from __future__ import annotations
import json, os
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
import vllm_gguf_plugin; vllm_gguf_plugin.register()
from vllm.engine.arg_utils import EngineArgs
from vllm.config import replace
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.platforms import current_platform
from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
SIDE=Path('/home/funboy/models/ds41/dspark-v41-mtp-2bc89ac')
TARGET='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix/DSV41-mixedq2-00001-of-00005.gguf'
TCFG='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix'
RANK=int(os.environ.get('DS41_COMPONENT_RANK','0'))
OUT=Path(os.environ.get('DS41_DSPARK_REALWEIGHT_OUT',str(ROOT/f'reports/DS41-Q2-001/attempt038-dspark-real/realweight-gate-rank{RANK}.json')))
REL=0.005; ABS=0.125

def metric(a,b):
    af=a.float(); bf=b.float(); d=af-bf; rn=float(bf.norm()); dn=float(d.norm())
    return {'exact':bool(torch.equal(a,b)),'different':int(torch.count_nonzero(a!=b)),
            'numel':a.numel(),'rel_l2':dn/rn if rn else (0.0 if dn==0 else None),
            'max_abs':float(d.abs().max()) if d.numel() else 0.0,
            'finite':bool(torch.isfinite(af).all() and torch.isfinite(bf).all())}

def e8m0(scale):
    u=scale.view(torch.uint8).to(torch.int32)
    if bool((u==255).any()): raise RuntimeError('E8M0 NaN code 255 in real scale')
    return torch.pow(torch.tensor(2.0,device=scale.device,dtype=torch.float32),u-127)

def dequant_mxfp8(w,s):
    sf=e8m0(s).repeat_interleave(32,0).repeat_interleave(32,1)
    return (w.float()*sf).to(torch.bfloat16)

_FP4=torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],dtype=torch.float32)
def dequant_mxfp4(packed,scale):
    u=packed.view(torch.uint8)
    tab=_FP4.to(u.device)
    lo=tab[(u & 0x0f).long()]; hi=tab[(u >> 4).long()]
    vals=torch.stack((lo,hi),dim=-1).flatten(-2)
    sf=e8m0(scale).repeat_interleave(32,-1)
    if vals.shape != sf.shape: raise RuntimeError(f'FP4 scale shape {sf.shape} != values {vals.shape}')
    return (vals*sf).to(torch.bfloat16)

def get(name):
    for p in sorted(SIDE.glob('model-0004[4-6]-of-00048.safetensors')):
        with safe_open(str(p),framework='pt',device='cpu') as f:
            if name in f.keys(): return f.get_tensor(name)
    raise KeyError(name)

def make_cfg():
    a=EngineArgs(model=TARGET,hf_config_path=TCFG,tokenizer=TCFG,config_format='gguf',load_format='gguf',quantization='gguf',dtype='bfloat16',tensor_parallel_size=1,enable_expert_parallel=False,max_model_len=4096,block_size=128,max_num_seqs=1,max_num_batched_tokens=1024,kv_cache_memory_bytes=1073741824,enable_prefix_caching=False,enable_chunked_prefill=True,enforce_eager=True,seed=1,speculative_config={'method':'dspark','model':str(SIDE),'num_speculative_tokens':1,'quantization':'fp8','enable_adaptive_verification':False,'draft_tensor_parallel_size':1,'draft_load_config':{'load_format':'safetensors','safetensors_load_strategy':'lazy'}})
    cfg=a.create_engine_config(usage_context=None); sp=cfg.speculative_config; q=get_draft_quant_config(cfg)
    return replace(cfg,model_config=sp.draft_model_config,load_config=sp.draft_load_config,quant_config=q,parallel_config=sp.draft_parallel_config),q

def main():
    assert torch.cuda.is_available(); torch.cuda.set_device(0)
    dcfg,q=make_cfg()
    report={'schema':'ds41-attempt038-realweight-gfx1151-v1','status':'RUNNING','rank':RANK,'device':torch.cuda.get_device_name(0),'gcn':getattr(torch.cuda.get_device_properties(0),'gcnArchName',None),'quant_class':type(q).__name__,'thresholds':{'rel_l2':REL,'max_abs':ABS},'linear':{},'expert':{}}
    assert type(q).__name__=='DeepseekV4FP8Config'
    # Real MXFP8 wq_a: execute the gfx1151-selected kernel directly.
    w=get('mtp.0.attn.wq_a.weight').cuda(); s=get('mtp.0.attn.wq_a.scale').cuda()
    class LinearProbe(torch.nn.Module): pass
    lin=LinearProbe().cuda()
    lin.weight=torch.nn.Parameter(w.clone(),requires_grad=False)
    # Runtime MXFP8 scale shape expands the 32-row checkpoint blocks to per-row scales.
    runtime_scale=s.view(torch.uint8).repeat_interleave(32,dim=0).contiguous()
    lin.weight_scale=torch.nn.Parameter(runtime_scale,requires_grad=False)
    kernel=init_mxfp8_linear_kernel()
    kernel_name=type(kernel).__name__
    kernel.process_weights_after_loading(lin)
    refw=dequant_mxfp8(w,s)
    wm=metric(lin.weight,refw)
    g=torch.Generator(device='cuda').manual_seed(38001+RANK); x=torch.randn((1,5120),device='cuda',dtype=torch.bfloat16,generator=g)
    y=kernel.apply_weights(lin,x); yref=F.linear(x,refw); ym=metric(y,yref)
    report['linear']={'source_weight_dtype':str(w.dtype),'source_scale_dtype':str(s.dtype),'kernel':kernel_name,'runtime_weight_dtype':str(lin.weight.dtype),'weight_vs_independent_dequant':wm,'output_vs_reference':ym,'cpu_fallback':False}
    if kernel_name!='EmulationMxfp8LinearKernel' or not wm['exact'] or not ym['finite'] or ym['rel_l2']>REL or ym['max_abs']>ABS: raise RuntimeError(f'MXFP8 gate failed {report["linear"]}')
    del lin,w,s,refw,y,yref
    torch.cuda.empty_cache()

    # Real expert0 MXFP4: exact checkpoint dequant -> BF16, then the real vLLM Triton fused-experts kernel.
    w1=get('mtp.0.ffn.experts.0.w1.weight').cuda(); s1=get('mtp.0.ffn.experts.0.w1.scale').cuda()
    w3=get('mtp.0.ffn.experts.0.w3.weight').cuda(); s3=get('mtp.0.ffn.experts.0.w3.scale').cuda()
    w2=get('mtp.0.ffn.experts.0.w2.weight').cuda(); s2=get('mtp.0.ffn.experts.0.w2.scale').cuda()
    dw1=dequant_mxfp4(w1,s1); dw3=dequant_mxfp4(w3,s3); dw2=dequant_mxfp4(w2,s2)
    w13=torch.stack((torch.cat((dw1,dw3),dim=0),),dim=0).contiguous()
    w2rt=dw2.unsqueeze(0).contiguous()
    g=torch.Generator(device='cuda').manual_seed(38101+RANK); xe=torch.randn((1,5120),device='cuda',dtype=torch.bfloat16,generator=g)*0.2
    ids=torch.zeros((1,1),device='cuda',dtype=torch.int32); tw=torch.ones((1,1),device='cuda',dtype=torch.float32)
    torch.cuda.synchronize()
    ye=fused_experts(xe,w13,w2rt,tw,ids,activation=MoEActivation.SILU,global_num_experts=1)
    torch.cuda.synchronize()
    gate=F.linear(xe,dw1); up=F.linear(xe,dw3)
    gate=torch.clamp(gate,max=7.0); up=torch.clamp(up,min=-7.0,max=7.0)
    act=gate*torch.sigmoid(gate)*up
    yeref=F.linear(act,dw2)
    em=metric(ye,yeref)
    report['expert']={'source_weight_dtype':str(w1.dtype),'source_scale_dtype':str(s1.dtype),'dequant_dtype':str(dw1.dtype),'backend':'TRITON_BF16_FUSED_EXPERTS','output_vs_independent_reference':em,'cpu_fallback':False,'packed_bytes':int(w1.numel()+w3.numel()+w2.numel()),'bf16_bytes':int((dw1.numel()+dw3.numel()+dw2.numel())*2)}
    if not em['finite'] or em['rel_l2']>REL or em['max_abs']>ABS: raise RuntimeError(f'MXFP4->BF16 expert gate failed {report["expert"]}')
    report['status']='PASS'; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
