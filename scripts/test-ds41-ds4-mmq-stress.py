#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes,json
from pathlib import Path
import numpy as np,torch,gguf
from _ds41_artifact import MODEL_DIR

def ptr(t): return ctypes.c_void_p(t.data_ptr())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fixture',required=True); ap.add_argument('--lib',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
 f=torch.load(a.fixture,weights_only=False); T=min(128,int(f['tokens'])); H=5120; M=2304; G=384; li=int(f['layer_index']); emc=f['expert_map'].long(); loc=sorted((int(l),g) for g,l in enumerate(emc.tolist()) if int(l)>=0); idx=np.asarray([g for _,g in loc],np.int64); L=len(idx)
 rs=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]; by={t.name:t for r in rs for t in r.tensors}; gt,ut,dt=[by[f'blk.{li}.{n}'] for n in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps')]
 gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).cuda(); up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).cuda(); down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).cuda(); em=f['expert_map'].cuda().int().contiguous(); rw=f['topk_weights'][:T].cuda().float().contiguous(); ids=f['topk_ids'][:T].cuda().int().contiguous()
 lib=ctypes.CDLL(a.lib,mode=ctypes.RTLD_GLOBAL); lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int; assert lib.ds4_mmq_init(0)==0; fn=lib.ds41_ds4_moe_full; P=ctypes.c_void_p; fn.argtypes=[P]*7+[ctypes.c_int]*5+[ctypes.c_float]+[P]*8; fn.restype=ctypes.c_int
 def run(x,ids0,rw0):
  s=(torch.empty((T,6),device='cuda',dtype=torch.int32),torch.empty((T,H),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,H),device='cuda'),torch.empty((T,H),device='cuda')); stream=P(torch.cuda.current_stream().cuda_stream); rc=fn(ptr(x),ptr(ids0),ptr(rw0),ptr(em),ptr(gate),ptr(up),ptr(down),T,H,M,G,L,ctypes.c_float(10.0),*[ptr(q) for q in s],stream); assert rc==0,rc; torch.cuda.synchronize(); return s[-1].to(torch.bfloat16).cpu(),s[0].cpu()
 zero=torch.zeros((T,H),device='cuda',dtype=torch.bfloat16); oz,lz=run(zero,ids,rw); zero_ok=bool(torch.equal(oz,torch.zeros_like(oz)))
 remote=torch.nonzero(f['expert_map']<0,as_tuple=False).flatten()[:6].to(torch.int32); assert remote.numel()==6; rid=remote.view(1,6).repeat(T,1).cuda(); rrw=torch.full((T,6),1/6,device='cuda'); ro,rl=run(f['x'][:T].cuda().contiguous(),rid,rrw); remote_ok=bool(torch.equal(ro,torch.zeros_like(ro))) and bool((rl<0).all())
 out={'schema':'ds41-ds4-mmq-stress-v1','status':'PASS' if zero_ok and remote_ok else 'FAIL','zero_activation_exact_zero':zero_ok,'remote_only_exact_zero':remote_ok,'remote_global_ids':remote.tolist(),'tokens':T}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
