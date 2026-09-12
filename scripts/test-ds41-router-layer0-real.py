#!/usr/bin/env python3
"""Independent layer0 V4.1 router check on real BF16/F32 GGUF weights."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from _ds41_artifact import MODEL_FILE
import torch
from gguf import GGUFReader
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias

MODEL=MODEL_FILE
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
TOPK=6
SCALE=1.5

def tensor(reader,name):
    t=next(t for t in reader.tensors if t.name==name)
    a=np.ascontiguousarray(t.data)
    if t.tensor_type.name=='BF16':
        return torch.from_numpy(a).view(torch.bfloat16).reshape(*(int(x) for x in reversed(t.shape)))
    if t.tensor_type.name=='F32':
        return torch.from_numpy(a).float().reshape(*(int(x) for x in reversed(t.shape)))
    raise TypeError((name,t.tensor_type))

def main():
    assert torch.cuda.is_available()
    r=GGUFReader(str(MODEL))
    w=tensor(r,'blk.0.ffn_gate_inp').to('cuda') # [384,5120] bf16
    bias=tensor(r,'blk.0.exp_probs_b').to('cuda') # [384] f32
    assert w.shape==(384,5120) and bias.shape==(384,)
    g=torch.Generator(device='cuda').manual_seed(411)
    x=torch.randn((8,5120),generator=g,device='cuda',dtype=torch.bfloat16)
    # Match the reference runtime: routing logits in float32 from BF16 model input/weight.
    logits=x.float() @ w.float().t()
    got_w,got_id=fused_topk_bias(
        hidden_states=x,
        gating_output=logits,
        scoring_func='sqrtsoftplus',
        e_score_correction_bias=bias,
        topk=TOPK,
        renormalize=True,
        indices_type=torch.int32,
        input_tokens=torch.arange(8,device='cuda',dtype=torch.int32)+100,
        hash_indices_table=None,
        routed_scaling_factor=SCALE,
        bias_vl=None,
        image_sentinel_lo=0,
    )
    # Independent formula from the pinned Vontra V4.1 runtime.
    ln=logits.detach().cpu().numpy().astype(np.float64)
    bb=bias.detach().cpu().numpy().astype(np.float64)
    scores=np.sqrt(np.logaddexp(ln,0.0))
    rank_scores=scores+bb[None,:]
    # top-k set; order must match descending selection as returned by vLLM.
    ref_id=np.argsort(rank_scores,axis=-1)[:,-TOPK:][:,::-1].copy()
    ref_w=np.take_along_axis(scores,ref_id,axis=-1)
    ref_w=ref_w/(ref_w.sum(axis=-1,keepdims=True)+1e-20)*SCALE
    gi=got_id.detach().cpu().numpy().astype(np.int64)
    gw=got_w.detach().cpu().numpy().astype(np.float64)
    ids_equal=np.array_equal(gi,ref_id)
    # If implementation chooses a different order among the same top-k, align by id.
    aligned=[]
    for row in range(len(gi)):
        d={int(i):float(v) for i,v in zip(ref_id[row],ref_w[row])}
        aligned.append([d[int(i)] for i in gi[row]])
    aligned=np.asarray(aligned,dtype=np.float64)
    diff=np.abs(gw-aligned)
    out={
      'status':'PASS' if ids_equal and float(diff.max())<2e-6 else 'FAIL',
      'ids_equal':bool(ids_equal),
      'max_weight_abs':float(diff.max()),
      'mean_weight_abs':float(diff.mean()),
      'weight_row_sums':got_w.float().sum(-1).detach().cpu().tolist(),
      'got_ids':gi.tolist(),
      'ref_ids':ref_id.tolist(),
      'logit_min':float(logits.min()),'logit_max':float(logits.max()),
      'bias_min':float(bias.min()),'bias_max':float(bias.max()),
    }
    p=ROOT/'reports/DS41-Q2-001/stage0/router-layer0-real-node01.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if out['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
