#!/usr/bin/env python3
"""K2 DSpark contract gate without model load.

Exercises the real shared DSpark/DFlash prepare-inputs Triton kernel for a K2
verify packet (target width=3) after 0/1/2 draft rejections and verifies native
acceptance accounting, including a one-draft tail.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.metrics.stats import RequestSpecDecodeMetrics
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

MASK_TOKEN=123
K=2

def run_case(*, positions, rejected, state_idx, last_token):
    device=torch.device('cuda')
    max_reqs=4; max_tokens=16
    nctx=len(positions)
    inp=SimpleNamespace(
        input_ids=torch.full((max_tokens,),-1,dtype=torch.int32,device=device),
        positions=torch.full((max_tokens,),-1,dtype=torch.int64,device=device),
        query_start_loc=torch.full((max_reqs+1,),-1,dtype=torch.int32,device=device),
        seq_lens=torch.full((max_reqs,),-1,dtype=torch.int32,device=device),
    )
    batch=SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([nctx],dtype=np.int32),
        positions=torch.tensor(positions,dtype=torch.int64,device=device),
        query_start_loc=torch.tensor([0,nctx],dtype=torch.int32,device=device),
        idx_mapping=torch.tensor([state_idx],dtype=torch.int32,device=device),
    )
    qslots=torch.full((max_tokens,),-2,dtype=torch.int64,device=device)
    cpos=torch.full((max_tokens,),-1,dtype=torch.int64,device=device)
    cslots=torch.full((max_tokens,),-2,dtype=torch.int64,device=device)
    sind=torch.full((max_reqs*K,),-1,dtype=torch.int64,device=device)
    spos=torch.full_like(sind,-1)
    smap=torch.full(sind.shape,-1,dtype=torch.int32,device=device)
    temp=torch.zeros(max_reqs,dtype=torch.float32,device=device)
    seeds=torch.zeros(max_reqs,dtype=torch.int64,device=device)
    itemp=torch.zeros(max_reqs,dtype=torch.float32,device=device); itemp[state_idx]=1.0
    iseeds=torch.zeros(max_reqs,dtype=torch.int64,device=device); iseeds[state_idx]=17+state_idx
    last=torch.zeros(max_reqs,dtype=torch.int64,device=device); last[state_idx]=last_token
    next_prefill=torch.zeros_like(last)
    # Enough non-null blocks for all positions.
    block_table=torch.tensor([[7,8,9,10,11,12,13,14]],dtype=torch.int32,device=device)
    prepare_dflash_inputs(
        inp,qslots,cpos,cslots,sind,spos,smap,temp,seeds,batch,
        torch.tensor([1],dtype=torch.int32,device=device),
        torch.tensor([rejected],dtype=torch.int32,device=device),
        last,next_prefill,itemp,iseeds,block_table,4,0,1,1,MASK_TOKEN,
        K,K,max_reqs,max_tokens,128,sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    valid=nctx-rejected
    assert valid>=1
    last_valid=positions[valid-1]
    expected_q=[last_valid+1,last_valid+2]
    got_q=inp.positions[:K].cpu().tolist()
    got_ids=inp.input_ids[:K].cpu().tolist()
    got_ctx=cpos[:nctx].cpu().tolist()
    got_cslots=cslots[:nctx].cpu().tolist()
    assert got_q==expected_q,(got_q,expected_q)
    assert got_ids==[last_token,MASK_TOKEN],got_ids
    assert sind[:K].cpu().tolist()==[0,1]
    assert spos[:K].cpu().tolist()==[expected_q[0]+1,expected_q[1]+1]
    assert smap[:K].cpu().tolist()==[state_idx,state_idx]
    assert got_ctx[:valid]==positions[:valid]
    assert got_ctx[valid:]==[0]*rejected
    assert got_cslots[valid:]==[PAD_SLOT_ID]*rejected
    return {
        'rejected':rejected,
        'accepted_drafts':K-rejected,
        'target_packet_positions':positions,
        'valid_context_positions':got_ctx[:valid],
        'query_positions':got_q,
        'query_input_ids':got_ids,
        'sample_positions':spos[:K].cpu().tolist(),
        'state_idx':state_idx,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--rank',type=int,required=True); a=ap.parse_args()
    assert torch.cuda.is_available(); torch.cuda.set_device(0)
    # A target verify packet for K2 has anchor + 2 drafted positions = T3.
    cases=[run_case(positions=[10,11,12],rejected=r,state_idx=0,last_token=99) for r in (0,1,2)]
    # A fresh request after full rejection must derive positions/state only from its own batch.
    fresh=run_case(positions=[20,21,22],rejected=0,state_idx=1,last_token=77)
    assert fresh['query_positions']==[23,24] and fresh['query_input_ids']==[77,MASK_TOKEN]

    # Native detailed metrics: distinguish zero/one/two accepts and a K-tail with only one proposal.
    m=RequestSpecDecodeMetrics.new(2)
    for drafted,accepted in ((2,0),(2,1),(2,2),(1,1)):
        m.observe(num_draft_tokens=drafted,num_accepted=accepted,detailed=True)
    md=m.to_dict()
    assert md['acceptance_histogram']==[1,2,1],md
    assert md['per_step_accepted']==[0,1,2,1]
    assert md['per_step_drafted']==[2,2,2,1]
    assert md['num_draft_tokens']==7 and md['num_accepted_draft_tokens']==4
    out={
        'schema':'ds41-dspark-k2-contract-v1','status':'PASS','rank':a.rank,
        'device':torch.cuda.get_device_name(0),
        'gcn':getattr(torch.cuda.get_device_properties(0),'gcnArchName',None),
        'K':2,'num_query_per_req':2,'target_verify_width_max':3,
        'cases':cases,'fresh_request':fresh,'tail_accounting':md,
        'target_dispatch_contract':'T3 uses already-qualified DS41_NATIVE_HIP_MOE_ROWWISE + DS41_MHC_ROWWISE_BLOCK; no target math change',
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
