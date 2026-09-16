#!/usr/bin/env python3
"""Bitwise M=1024 gate for the one-shot DS41 V2 prefill profiler wrappers."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from runtime.ds41.perf_profile import DS41PerfCollector
import vllm.models.deepseek_v4_1.amd.model as model_mod
M=1024; H=5120; HC=4; MIX=24

def main():
    g=torch.Generator(device='cuda').manual_seed(551024)
    res=torch.randn((M,HC,H),device='cuda',dtype=torch.bfloat16,generator=g)
    fn=(torch.randn((MIX,HC*H),device='cuda',dtype=torch.float32,generator=g)*0.002).contiguous()
    scale=torch.tensor([0.7,0.8,0.3],device='cuda',dtype=torch.float32)
    base=(torch.randn((MIX,),device='cuda',dtype=torch.float32,generator=g)*0.01).contiguous()
    pre=torch.sigmoid(torch.randn((M,HC),device='cuda',dtype=torch.float32,generator=g)).contiguous()
    args=(res,fn,scale,base,1e-20,1e-6,1e-6,2.0,20)
    c=DS41PerfCollector(); c.install()
    ref=tuple(x.detach().clone() for x in model_mod.mhc_pre_delayed_torch(*args,pre_mix=pre))
    c.enable(); cand=tuple(x.detach().clone() for x in model_mod.mhc_pre_delayed_torch(*args,pre_mix=pre)); c.disable(); torch.cuda.synchronize()
    pre_exact=all(torch.equal(a,b) for a,b in zip(ref,cand,strict=True))
    x=torch.randn((M,H),device='cuda',dtype=torch.bfloat16,generator=g)
    post_ref=model_mod.mhc_post_torch(x,res,ref[0],ref[1]).detach().clone()
    c.enable(); post_cand=model_mod.mhc_post_torch(x,res,ref[0],ref[1]).detach().clone(); c.disable(); torch.cuda.synchronize()
    post_exact=torch.equal(post_ref,post_cand)
    out={'schema':'ds41-prefill-profile-equivalence-v1','status':'PASS' if pre_exact and post_exact else 'FAIL','tokens':M,'pre_bitwise':pre_exact,'post_bitwise':post_exact}
    path=Path('runtime/ds41/results/prefill-profile-v2-gate.json'); path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out)); raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__': main()
