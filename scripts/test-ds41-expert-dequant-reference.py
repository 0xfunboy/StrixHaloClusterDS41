#!/usr/bin/env python3
"""Compare one real MixedQ2 expert GPU GGUF kernels against independent gguf-py dequantization."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import gguf
from gguf.quants import dequantize
from vllm_gguf_plugin.quantization import fused_mul_mat_gguf
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation, ApplyMoEActivationConfig

MODEL=Path('/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2/DSV41-mixedq2-00001-of-00005.gguf')
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
EXPERT=0; LIMIT=10.0

def get(reader,name):
    t=next(t for t in reader.tensors if t.name==name)
    raw=np.ascontiguousarray(t.data[EXPERT])
    q=t.tensor_type
    dense=dequantize(raw,q).astype(np.float32,copy=False)
    return t,raw,dense

def metric(a,b):
    d=(a.float()-b.float()).abs(); denom=b.float().abs().clamp_min(1e-5)
    return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rmse':float(torch.sqrt((d*d).mean())),'max_rel_gt1e5':float((d/denom).max())}

def main():
    assert torch.cuda.is_available()
    r=gguf.GGUFReader(str(MODEL))
    tg,rg,wg=get(r,'blk.0.ffn_gate_exps')
    tu,ru,wu=get(r,'blk.0.ffn_up_exps')
    td,rd,wd=get(r,'blk.0.ffn_down_exps')
    assert wg.shape==(2304,5120),wg.shape
    assert wu.shape==(2304,5120),wu.shape
    assert wd.shape==(5120,2304),wd.shape
    qg=int(tg.tensor_type); qd=int(td.tensor_type)
    wg_gpu=torch.from_numpy(rg).to('cuda'); wu_gpu=torch.from_numpy(ru).to('cuda'); w13_gpu=torch.cat((wg_gpu,wu_gpu),dim=0)
    wd_gpu=torch.from_numpy(rd).to('cuda')
    # Real embedding-derived normalized inputs plus random normalized inputs.
    te=next(t for t in r.tensors if t.name=='token_embd') if any(t.name=='token_embd' for t in r.tensors) else None
    # token_embd is in shard5, so use deterministic normalized test inputs here; they match layer RMS scale.
    gen=torch.Generator(device='cuda').manual_seed(412)
    x=torch.randn((4,5120),generator=gen,device='cuda',dtype=torch.bfloat16)
    x=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-20)).to(torch.bfloat16)

    pre_gpu=fused_mul_mat_gguf(x,w13_gpu,qg)
    W13=torch.from_numpy(np.concatenate([wg,wu],axis=0)).to('cuda')
    pre_ref=(x.float() @ W13.t()).to(torch.bfloat16)
    pre_m=metric(pre_gpu,pre_ref)

    d=pre_gpu.shape[-1]//2
    act_gpu=torch.empty(pre_gpu.shape[:-1]+(d,),dtype=pre_gpu.dtype,device='cuda')
    apply_moe_activation(MoEActivation.SILU,act_gpu,pre_gpu,activation_config=ApplyMoEActivationConfig(clamp_limit=LIMIT))
    # Independent explicit formula on dequant-reference preactivation (float32), rounded to BF16 as kernel path.
    pr=pre_ref.float(); gate=torch.clamp(pr[...,:d],max=LIMIT); up=torch.clamp(pr[...,d:],min=-LIMIT,max=LIMIT)
    act_ref=(gate*torch.sigmoid(gate)*up).to(torch.bfloat16)
    act_m=metric(act_gpu,act_ref)

    down_gpu=fused_mul_mat_gguf(act_gpu,wd_gpu,qd)
    WD=torch.from_numpy(wd).to('cuda')
    down_ref=(act_ref.float() @ WD.t()).to(torch.bfloat16)
    down_m=metric(down_gpu,down_ref)

    pg=pre_gpu.float(); gate_pg=pg[...,:d]; up_pg=pg[...,d:]
    out={
      'status':'PASS', 'expert':EXPERT,
      'qtypes':{'gate_up':tg.tensor_type.name,'down':td.tensor_type.name},
      'shapes':{'gate':list(wg.shape),'up':list(wu.shape),'down':list(wd.shape)},
      'preactivation':pre_m,
      'activation':act_m,
      'down':down_m,
      'clamp_activity':{'gate_gt10':int((gate_pg>LIMIT).sum()),'up_abs_gt10':int((up_pg.abs()>LIMIT).sum()),'gate_max':float(gate_pg.max()),'up_abs_max':float(up_pg.abs().max())},
      'norms':{'down_gpu':float(down_gpu.float().norm()),'down_ref':float(down_ref.float().norm())},
    }
    # Low-bit kernels are approximate matmuls over identical decoded weights; tolerate BF16-scale accumulation error,
    # but fail on orientation/scale interpretation or large output divergence.
    if pre_m['rmse']>0.08 or down_m['rmse']>0.12 or down_m['max_abs']>2.0:
        out['status']='FAIL'
    p=ROOT/'reports/DS41-Q2-001/stage0/expert0-dequant-reference.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if out['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
