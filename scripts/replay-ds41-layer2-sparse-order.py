#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

def met(a,b):
 d=a.float()-b.float(); return {'exact':bool(torch.equal(a,b)),'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'rel_l2':float(d.norm())/max(float(b.float().norm()),1e-30)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--out',required=True); q=ap.parse_args()
 from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_prefill_triton
 A=torch.load(q.a,weights_only=False,map_location='cpu')['layer2_attention']['1']; B=torch.load(q.b,weights_only=False,map_location='cpu')['layer2_attention']['1']
 qa=A['q_final'].cuda().unsqueeze(0); qb=B['q_final'].cuda().unsqueeze(0); assert torch.equal(qa,qb)
 sink=A['attn_sink'].cuda(); assert torch.equal(A['attn_sink'],B['attn_sink'])
 ia=A['valid_context_indices'].to(torch.int64); ib=B['valid_context_indices'].to(torch.int64)
 ra=A['context_rows']; rb=B['context_rows']; assert set(ia.tolist())==set(ib.tolist()) and len(set(ia.tolist()))==len(ia)
 # Align B to A logical workspace indices; prove row content is exact.
 posb={int(k):j for j,k in enumerate(ib.tolist())}; rb_aligned=torch.stack([rb[posb[int(k)]] for k in ia.tolist()]); assert torch.equal(ra,rb_aligned)
 def run(rows):
  kv=rows.cuda().unsqueeze(1).contiguous(); idx=torch.arange(rows.shape[0],device='cuda',dtype=torch.int32).view(1,-1); lens=torch.tensor([rows.shape[0]],device='cuda',dtype=torch.int32)
  y=_rocm_sparse_attn_prefill_triton(qa,kv.squeeze(1),idx,float(A['scale']),sink,448,64,lens)
  torch.cuda.synchronize(); return y.cpu()[0]
 ya=run(ra); yb=run(rb)
 # Canonical context: sort by logical workspace index, keeping the corresponding row.
 order_a=torch.argsort(ia,stable=True); order_b=torch.argsort(ib,stable=True)
 ca=ra[order_a]; cb=rb[order_b]; assert torch.equal(ca,cb)
 yc_a=run(ca); yc_b=run(cb)
 out={
  'schema':'ds41-layer2-sparse-order-replay-v1',
  'q_exact':True,'context_set_equal':True,'context_aligned_exact':True,'canonical_rows_exact':True,
  'captured_kernel_A_vs_B':met(A['kernel_output_final'],B['kernel_output_final']),
  'replay_A_vs_captured_A':met(ya,A['kernel_output_final']),
  'replay_B_vs_captured_B':met(yb,B['kernel_output_final']),
  'replay_A_vs_B':met(ya,yb),
  'canonical_A_vs_B':met(yc_a,yc_b),
  'replay_A_vs_canonical':met(ya,yc_a),
  'replay_B_vs_canonical':met(yb,yc_b),
  'num_context_rows':int(ra.shape[0]),
 }
 Path(q.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
