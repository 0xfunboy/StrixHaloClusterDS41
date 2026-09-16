#!/usr/bin/env python3
"""DS41 EP skip-remote correctness/timing gate at real prefill shapes.

Runs on real layer0 MixedQ2 expert bytes. For M=65 and M=1024 it compares a
full four-expert reference with the sum of two EP rank-local contributions.
The source under test must already contain the candidate skip-remote guard.
Timing is diagnostic only; correctness is exact max_abs=0/mean_abs=0.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import gguf
import numpy as np
import torch
from _ds41_artifact import MODEL_FILE
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod
IQ2_XXS=int(gguf.GGMLQuantizationType.IQ2_XXS); Q2_K=int(gguf.GGMLQuantizationType.Q2_K)
def raw(reader,name,experts):
    t=next(t for t in reader.tensors if t.name==name)
    return torch.from_numpy(np.ascontiguousarray(t.data[experts])).to('cuda')
def layer(w13,w2,expert_map):
    return SimpleNamespace(apply_router_weight_on_input=False,w13_weight=w13,w2_weight=w2,w13_weight_type=SimpleNamespace(weight_type=IQ2_XXS),w2_weight_type=SimpleNamespace(weight_type=Q2_K),activation=SimpleNamespace(value='silu'),expert_map=expert_map)
def method(): return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=10.0))
def timed(fn):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record(); y=fn(); e.record(); e.synchronize(); return y,float(s.elapsed_time(e))
def run(m):
    reader=gguf.GGUFReader(str(MODEL_FILE)); gate=raw(reader,'blk.0.ffn_gate_exps',slice(0,4)); up=raw(reader,'blk.0.ffn_up_exps',slice(0,4)); down=raw(reader,'blk.0.ffn_down_exps',slice(0,4)); w13=torch.cat((gate,up),dim=1); del gate,up
    g=torch.Generator(device='cuda').manual_seed(41+m); x=torch.randn((m,5120),generator=g,device='cuda',dtype=torch.bfloat16); idx=torch.arange(m,device='cuda',dtype=torch.int64)%4; ids=torch.stack((idx,(idx+1)%4),dim=1); wt=torch.tensor([0.625,0.375],device='cuda',dtype=torch.float32).expand(m,2).contiguous(); map0=torch.tensor([0,1,-1,-1],device='cuda',dtype=torch.int32); map1=torch.tensor([-1,-1,0,1],device='cuda',dtype=torch.int32)
    meth=method(); full_l=layer(w13,down,None); l0=layer(w13[:2],down[:2],map0); l1=layer(w13[2:],down[2:],map1)
    for L in (full_l,l0,l1): GGUFMoEMethod.apply(meth,L,x,wt,ids,None,None)
    torch.cuda.synchronize(); full,full_ms=timed(lambda:GGUFMoEMethod.apply(meth,full_l,x,wt,ids,None,None)); r0,r0_ms=timed(lambda:GGUFMoEMethod.apply(meth,l0,x,wt,ids,None,None)); r1,r1_ms=timed(lambda:GGUFMoEMethod.apply(meth,l1,x,wt,ids,None,None)); diff=((r0+r1).float()-full.float()).abs(); assert float(diff.max())==0.0 and float(diff.mean())==0.0
    return {'m':m,'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'full_ms':full_ms,'rank0_ms':r0_ms,'rank1_ms':r1_ms,'ep_critical_ms':max(r0_ms,r1_ms)}
def main():
    assert torch.cuda.is_available(); cases=[run(65),run(1024)]; print(json.dumps({'schema':'ds41-ep-prefill-skip-remote-gate-v1','status':'PASS','cases':cases},sort_keys=True))
if __name__=='__main__': main()
