#!/usr/bin/env python3
"""Offline CPU reference check for saved DS41 real sparse-attention packets.

Reference math is copied from vendored vLLM
`tests/kernels/attention/test_rocm_triton_attn_dsv4.py::_ref_sparse_prefill_ragged`.
No model is loaded and no GPU kernel is launched.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

ATOL = 2e-2
RTOL = 2e-2
REFERENCE_SOURCE = ".vendor/vllm-dsv41/tests/kernels/attention/test_rocm_triton_attn_dsv4.py::_ref_sparse_prefill_ragged"


def ref_one(q: torch.Tensor, kv: torch.Tensor, scale: float, sink: torch.Tensor | None) -> torch.Tensor:
    # q [H,D], kv [K,D]. Same FP32 score/softmax/value accumulation contract as upstream ref.
    qf=q.float(); kvf=kv.float(); out=torch.empty_like(qf)
    for h in range(q.shape[0]):
        scores=torch.mv(kvf,qf[h])*float(scale)
        if sink is not None:
            probs=torch.softmax(torch.cat([scores,sink[h].float().reshape(1)]),dim=0)[:-1]
        else:
            probs=torch.softmax(scores,dim=0)
        out[h]=torch.sum(probs[:,None]*kvf,dim=0)
    return out.to(torch.bfloat16)


def metric(actual: torch.Tensor, ref: torch.Tensor) -> dict:
    a=actual.float(); r=ref.float(); d=(a-r).abs()
    tol=ATOL + RTOL*r.abs()
    close=torch.isfinite(a)&torch.isfinite(r)&(d<=tol)
    rn=float(r.norm())
    return {
        "shape":list(actual.shape),"finite":bool(torch.isfinite(a).all() and torch.isfinite(r).all()),
        "allclose_upstream_gate":bool(close.all()),"not_close":int((~close).sum()),"numel":actual.numel(),
        "max_abs":float(d.max()),"mean_abs":float(d.mean()),"rel_l2":float((a-r).norm())/max(rn,1e-30),
        "atol":ATOL,"rtol":RTOL,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); ap.add_argument('--rank',type=int,required=True); q=ap.parse_args()
    pack=torch.load(q.input,map_location='cpu',weights_only=False)
    rows=[]
    for chunk in (0,1):
        x=pack['layer2_attention'][chunk]
        ref=ref_one(x['q_final'].cpu(),x['context_rows'].cpu(),float(x['scale']),x['attn_sink'].cpu())
        m=metric(x['kernel_output_final'].cpu(),ref)
        rows.append({"chunk":chunk,"position_start":x['position_start'],"position_end":x['position_end'],"context_rows":int(x['context_rows'].shape[0]),"metric":m})
    status='PASS' if all(r['metric']['allclose_upstream_gate'] for r in rows) else 'FAIL'
    out={"schema":"ds41-sparse-attn-real-reference-v1","status":status,"rank":q.rank,"input":q.input,"prompt_tokens":pack.get('prompt_tokens'),"request_index":pack.get('request_index'),"reference_source":REFERENCE_SOURCE,"gate":{"atol":ATOL,"rtol":RTOL,"source":"vendored upstream test assertion"},"chunks":rows,"model_loaded":False,"generation_requests":0,"gpu_kernel_launched":False}
    Path(q.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
    raise SystemExit(0 if status=='PASS' else 1)
if __name__=='__main__': main()
