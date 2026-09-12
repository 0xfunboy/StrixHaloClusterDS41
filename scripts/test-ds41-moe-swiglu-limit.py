#!/usr/bin/env python3
"""Prove GGUF routed MoE honors V4.1 swiglu_limit on real MixedQ2 expert bytes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import gguf
from gguf.quants import dequantize
from vllm_gguf_plugin.quantization import fused_moe_gguf

MODEL=Path('/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2/DSV41-mixedq2-00001-of-00005.gguf')
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
LIMIT=10.0

def get(reader,name):
    t=next(t for t in reader.tensors if t.name==name)
    raw=np.ascontiguousarray(t.data[0])
    dense=dequantize(raw,t.tensor_type).astype(np.float32,copy=False)
    return t,raw,dense

def metric(a,b):
    d=(a.float()-b.float()).abs()
    return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rmse':float(torch.sqrt((d*d).mean()))}

def main():
    assert torch.cuda.is_available()
    r=gguf.GGUFReader(str(MODEL))
    tg,rg,wg=get(r,'blk.0.ffn_gate_exps'); tu,ru,wu=get(r,'blk.0.ffn_up_exps'); td,rd,wd=get(r,'blk.0.ffn_down_exps')
    w13=torch.cat((torch.from_numpy(rg),torch.from_numpy(ru)),dim=0).unsqueeze(0).to('cuda')
    w2=torch.from_numpy(rd).unsqueeze(0).to('cuda')
    q1=int(tg.tensor_type); q2=int(td.tensor_type)
    gen=torch.Generator(device='cuda').manual_seed(413)
    base=torch.randn((2,5120),generator=gen,device='cuda',dtype=torch.bfloat16)
    base=(base.float()*torch.rsqrt(base.float().square().mean(-1,keepdim=True)+1e-20)).to(torch.bfloat16)
    # Scale deliberately to exercise the checkpoint's required clamp semantics.
    x=(base.float()*3.0).to(torch.bfloat16)
    ids=torch.zeros((2,1),device='cuda',dtype=torch.int64)
    weights=torch.ones((2,1),device='cuda',dtype=torch.float32)
    fixed=fused_moe_gguf(x,w13,w2,weights,ids,q1,q2,'silu',LIMIT)
    legacy=fused_moe_gguf(x,w13,w2,weights,ids,q1,q2,'silu',0.0)

    W1=torch.from_numpy(wg).to('cuda'); W3=torch.from_numpy(wu).to('cuda'); W2=torch.from_numpy(wd).to('cuda')
    gate=x.float()@W1.t(); up=x.float()@W3.t()
    gate_c=torch.clamp(gate,max=LIMIT); up_c=torch.clamp(up,min=-LIMIT,max=LIMIT)
    act=gate_c*torch.sigmoid(gate_c)*up_c
    ref=(act@W2.t()).to(torch.bfloat16)

    change=metric(fixed,legacy); fixed_ref=metric(fixed,ref); legacy_ref=metric(legacy,ref)
    out={
      'status':'PASS',
      'input_scale':3.0,
      'gate_max':float(gate.max()),'up_abs_max':float(up.abs().max()),
      'gate_gt10':int((gate>LIMIT).sum()),'up_abs_gt10':int((up.abs()>LIMIT).sum()),
      'fixed_vs_legacy':change,
      'fixed_vs_reference':fixed_ref,
      'legacy_vs_reference':legacy_ref,
      'fixed_norm':float(fixed.float().norm()),'legacy_norm':float(legacy.float().norm()),'reference_norm':float(ref.float().norm()),
    }
    # Causal gate: clamp must be exercised, fixed output must differ materially from legacy,
    # and must improve agreement with the independent dequantized reference.
    if out['gate_gt10']+out['up_abs_gt10']==0 or change['rmse']<0.05 or not (fixed_ref['rmse'] < legacy_ref['rmse']):
        out['status']='FAIL'
    path=ROOT/'reports/DS41-Q2-001/stage0/moe-swiglu-limit-real.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if out['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
