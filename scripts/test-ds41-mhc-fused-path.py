#!/usr/bin/env python3
"""End-to-end mHC-pre toggle gate for DS41 fused coefficient/Sinkhorn M=1."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch

import vllm.models.deepseek_v4_1.amd.model as model_mod

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
OUT = ROOT / 'reports/DS41-Q2-001/perf/mhc-fused-path-gate.json'
ATOL_FP32 = 5e-6
RTOL_FP32 = 5e-5
ATOL_POST = 2e-3
RTOL_POST = 2e-4


def metric(a, b):
    d = a.float() - b.float(); rn=float(b.float().norm())
    return {'max_abs':float(d.abs().max()), 'mean_abs':float(d.abs().mean()), 'rel_l2':float(d.norm())/max(rn,1e-30)}


def run_pair(label, residual, fn, scale, base, *, pre_mix=None, x=None):
    args=(residual,fn,scale,base,1e-20,1e-6,1e-6,2.0,20)
    os.environ['DS41_MHC_COEFF_SINKHORN']='0'
    ref=tuple(t.detach().clone() for t in model_mod.mhc_pre_delayed_torch(*args,pre_mix=pre_mix,x=x))
    os.environ['DS41_MHC_COEFF_SINKHORN']='1'
    cand=tuple(t.detach().clone() for t in model_mod.mhc_pre_delayed_torch(*args,pre_mix=pre_mix,x=x))
    rows={}
    for name,a,b in zip(('post_mix','comb_mix','layer_input','next_pre'),cand,ref,strict=True):
        if a.dtype == torch.float32:
            torch.testing.assert_close(a,b,atol=ATOL_FP32,rtol=RTOL_FP32)
        else:
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        rows[name]=metric(a,b)
    probe=torch.randn((residual.shape[0],residual.shape[-1]),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(811))
    ref_post=model_mod.mhc_post_torch(probe,residual,ref[0],ref[1])
    cand_post=model_mod.mhc_post_torch(probe,residual,cand[0],cand[1])
    torch.testing.assert_close(cand_post,ref_post,atol=ATOL_POST,rtol=RTOL_POST)
    rows['post_result']=metric(cand_post,ref_post)
    return {'label':label,'tokens':int(residual.shape[0]),'status':'PASS','metrics':rows}


def real_first_layer_case():
    p=ROOT/'scripts/test-ds41-layer0-complete-densefix.py'
    spec=importlib.util.spec_from_file_location('ds41_l0_gate',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    prompt=json.loads(mod.PROMPT.read_text())['prompts']['arithmetic']['token_ids']
    ids=torch.tensor([prompt[-1]],dtype=torch.long,device='cuda')
    emb=mod.bf16('token_embd')[ids]
    residual=emb[:,None,:].expand(-1,4,-1).contiguous()
    fn=mod.f32('blk.0.hc_attn_fn').reshape(24,4,5120).sum(1).contiguous()
    return run_pair('real-layer0-last-token-first-attn',residual,fn,mod.f32('blk.0.hc_attn_scale').contiguous(),mod.f32('blk.0.hc_attn_base').contiguous(),x=emb)


def random_regular(tokens):
    g=torch.Generator(device='cuda').manual_seed(18800+tokens)
    residual=torch.randn((tokens,4,5120),device='cuda',dtype=torch.bfloat16,generator=g)
    fn=(torch.randn((24,20480),device='cuda',dtype=torch.float32,generator=g)*0.001).contiguous()
    scale=torch.tensor([0.7,0.8,0.3],device='cuda',dtype=torch.float32)
    base=(torch.randn((24,),device='cuda',dtype=torch.float32,generator=g)*0.01).contiguous()
    pre_mix=torch.sigmoid(torch.randn((tokens,4),device='cuda',dtype=torch.float32,generator=g)).contiguous()
    return run_pair(f'random-delayed-M{tokens}',residual,fn,scale,base,pre_mix=pre_mix)


def main():
    assert torch.cuda.is_available()
    rows=[real_first_layer_case(),random_regular(1),random_regular(7)]
    # M>1 must remain exact fallback even when the environment flag is enabled.
    assert rows[-1]['metrics']['post_mix']['max_abs'] == 0.0
    assert rows[-1]['metrics']['comb_mix']['max_abs'] == 0.0
    out={'status':'PASS','tolerances':{'fp32_atol':ATOL_FP32,'fp32_rtol':RTOL_FP32,'post_atol':ATOL_POST,'post_rtol':RTOL_POST},'cases':rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
    os.environ.pop('DS41_MHC_COEFF_SINKHORN',None)

if __name__=='__main__': main()
