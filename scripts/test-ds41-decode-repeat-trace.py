#!/usr/bin/env python3
from __future__ import annotations
import json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def batch(req,before,plen,ids,pos,drafts=0):
    ids=torch.tensor(ids,dtype=torch.int64); pos=torch.tensor(pos,dtype=torch.int64)
    n=len(ids); li=torch.arange(n,dtype=torch.int64)
    return SimpleNamespace(req_ids=[req],cu_num_logits_np=np.array([0,n],np.int32),logits_indices=li,
      input_ids=ids,positions=pos,expanded_local_pos=torch.arange(n,dtype=torch.int32),
      num_computed_tokens_np=np.array([before],np.int32),num_computed_prefill_tokens_np=np.array([min(before,plen)],np.int32),
      prefill_len_np=np.array([plen],np.int32),is_prefilling_np=np.array([before<plen]),
      num_scheduled_tokens=np.array([n],np.int32),num_draft_tokens_per_req=np.array([drafts],np.int32),
      idx_mapping_np=np.array([0],np.intp))

def samp(vals,ns,nr):
    return SimpleNamespace(sampled_token_ids=torch.tensor([vals],dtype=torch.int64),num_sampled=torch.tensor([ns]),num_rejected=torch.tensor([nr]))
with tempfile.TemporaryDirectory() as td:
 os.environ['DS41_DECODE_TRACE_DIR']=td; os.environ['DS41_DECODE_TRACE_EMIT_CAP']='4'; os.environ['RANK']='0'
 import runtime.ds41.decode_repeat_diag as m; m._TRACE=None; t=m.get_decode_repeat_trace()
 # Wrong prompt does not arm.
 t.begin_step(batch('x',0,9,[1],[0])); assert t.current is None
 # Prefill first sample.
 b=batch('r',0,1588,[10],[1587]); t.begin_step(b); assert t.current is not None
 t.record_raw_logits(b,torch.randn(1,16),[False]); t.record_sampler(b,samp([7,0,0],1,0))
 # Decode verify step: three target rows, two drafts, top-k at source layer.
 b=batch('r',1589,1588,[7,8,9],[1588,1589,1590],2); t.begin_step(b)
 topk=torch.tensor([[5,2,-1],[7,1,-1],[9,3,-1]],dtype=torch.int32)
 meta=SimpleNamespace(num_decode_tokens=3,num_prefill_tokens=0)
 t.record_indexer(2,b.positions,meta,topk)
 q=torch.randn(3,2,4); swa=torch.tensor([[0,1],[1,2],[2,3]],dtype=torch.int32); lens=torch.tensor([2,2,2],dtype=torch.int32)
 t.record_decode_attention(2,q,swa,lens,torch.tensor([4,5,6],dtype=torch.int32),torch.tensor([0,1,2,3],dtype=torch.int32),torch.tensor([1,1,1],dtype=torch.int32))
 t.record_decode_attention_output(2,torch.randn(3,2,4)); t.record_raw_logits(b,torch.randn(3,16),[False]); t.record_sampler(b,samp([8,9,0],2,1))
 # Third step reaches cap and finalizes.
 b=batch('r',1591,1588,[9,11],[1591,1592],1); t.begin_step(b); t.record_raw_logits(b,torch.randn(2,16),[False]); t.record_sampler(b,samp([11,12,0],2,0))
 assert t.current is None and t.completed==1
 p=Path(td)/'decode-trace-rank0-request0.pt'; j=Path(td)/'decode-trace-rank0-request0.json'; assert p.exists() and j.exists()
 d=torch.load(p,weights_only=False); assert d['emitted_count']==5 and len(d['steps'])==3
 assert d['steps'][1]['indexer']['2']['num_decode_tokens']==3
 assert d['steps'][1]['target_positions'].tolist()==[1588,1589,1590]
 assert json.load(open(j))['status']=='COMPLETE'
print('DECODE_REPEAT_TRACE_MODEL_FREE=PASS')
