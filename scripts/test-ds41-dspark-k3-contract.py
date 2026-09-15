#!/usr/bin/env python3
"""K3 DSpark contract gate without full model load.

Exercises the real shared DSpark prepare-inputs Triton kernel for target width T4
through 0/1/2/3 rejections, fresh-request state, and native detailed acceptance
accounting with full-width plus K2/K1 tails.
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
K=3

def run_case(*, positions, rejected, state_idx, last_token):
    device=torch.device('cuda'); max_reqs=4; max_tokens=24; nctx=len(positions)
    inp=SimpleNamespace(
        input_ids=torch.full((max_tokens,),-1,dtype=torch.int32,device=device),
        positions=torch.full((max_tokens,),-1,dtype=torch.int64,device=device),
        query_start_loc=torch.full((max_reqs+1,),-1,dtype=torch.int32,device=device),
        seq_lens=torch.full((max_reqs,),-1,dtype=torch.int32,device=device),
    )
    batch=SimpleNamespace(
        num_reqs=1,num_scheduled_tokens=np.array([nctx],dtype=np.int32),
        positions=torch.tensor(positions,dtype=torch.int64,device=device),
        query_start_loc=torch.tensor([0,nctx],dtype=torch.int32,device=device),
        idx_mapping=torch.tensor([state_idx],dtype=torch.int32,device=device),
    )
    qslots=torch.full((max_tokens,),-2,dtype=torch.int64,device=device)
    cpos=torch.full((max_tokens,),-1,dtype=torch.int64,device=device)
    cslots=torch.full((max_tokens,),-2,dtype=torch.int64,device=device)
    sind=torch.full((max_reqs*K,),-1,dtype=torch.int64,device=device); spos=torch.full_like(sind,-1)
    smap=torch.full(sind.shape,-1,dtype=torch.int32,device=device)
    temp=torch.zeros(max_reqs,dtype=torch.float32,device=device); seeds=torch.zeros(max_reqs,dtype=torch.int64,device=device)
    itemp=torch.zeros(max_reqs,dtype=torch.float32,device=device); itemp[state_idx]=1.0
    iseeds=torch.zeros(max_reqs,dtype=torch.int64,device=device); iseeds[state_idx]=31+state_idx
    last=torch.zeros(max_reqs,dtype=torch.int64,device=device); last[state_idx]=last_token
    next_prefill=torch.zeros_like(last)
    block_table=torch.tensor([[7,8,9,10,11,12,13,14]],dtype=torch.int32,device=device)
    prepare_dflash_inputs(
        inp,qslots,cpos,cslots,sind,spos,smap,temp,seeds,batch,
        torch.tensor([1],dtype=torch.int32,device=device),
        torch.tensor([rejected],dtype=torch.int32,device=device),
        last,next_prefill,itemp,iseeds,block_table,4,0,1,1,MASK_TOKEN,
        K,K,max_reqs,max_tokens,128,sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    valid=nctx-rejected; assert valid>=1
    last_valid=positions[valid-1]; expected_q=[last_valid+i for i in (1,2,3)]
    got_q=inp.positions[:K].cpu().tolist(); got_ids=inp.input_ids[:K].cpu().tolist()
    got_ctx=cpos[:nctx].cpu().tolist(); got_cslots=cslots[:nctx].cpu().tolist()
    assert got_q==expected_q,(got_q,expected_q)
    assert got_ids==[last_token,MASK_TOKEN,MASK_TOKEN],got_ids
    assert sind[:K].cpu().tolist()==[0,1,2]
    assert spos[:K].cpu().tolist()==[x+1 for x in expected_q]
    assert smap[:K].cpu().tolist()==[state_idx]*K
    assert got_ctx[:valid]==positions[:valid]
    assert got_ctx[valid:]==[0]*rejected
    assert got_cslots[valid:]==[PAD_SLOT_ID]*rejected
    return {'rejected':rejected,'accepted_drafts':K-rejected,'target_packet_positions':positions,'valid_context_positions':got_ctx[:valid],'query_positions':got_q,'query_input_ids':got_ids,'sample_positions':spos[:K].cpu().tolist(),'state_idx':state_idx}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--rank',type=int,required=True); a=ap.parse_args()
    assert torch.cuda.is_available(); torch.cuda.set_device(0)
    # target packet = authoritative anchor/bonus + 3 draft positions => T4.
    cases=[run_case(positions=[10,11,12,13],rejected=r,state_idx=0,last_token=99) for r in (0,1,2,3)]
    fresh=run_case(positions=[20,21,22,23],rejected=0,state_idx=1,last_token=77)
    assert fresh['query_positions']==[24,25,26] and fresh['query_input_ids']==[77,MASK_TOKEN,MASK_TOKEN]
    m=RequestSpecDecodeMetrics.new(3)
    # Full blocks cover 0/1/2/3 accepted. Last two rows model residual K2/K1 tails.
    for drafted,accepted in ((3,0),(3,1),(3,2),(3,3),(2,2),(1,1)):
        m.observe(num_draft_tokens=drafted,num_accepted=accepted,detailed=True)
    md=m.to_dict()
    assert md['acceptance_histogram']==[1,2,2,1],md
    assert md['per_step_accepted']==[0,1,2,3,2,1]
    assert md['per_step_drafted']==[3,3,3,3,2,1]
    assert md['num_draft_tokens']==15 and md['num_accepted_draft_tokens']==9
    out={'schema':'ds41-dspark-k3-contract-v1','status':'PASS','rank':a.rank,'device':torch.cuda.get_device_name(0),'gcn':getattr(torch.cuda.get_device_properties(0),'gcnArchName',None),'K':3,'num_query_per_req':3,'target_verify_width_max':4,'scheduler_additional_slots':2,'mtp_stages':3,'target_hidden_layers':[37,38,39],'cases':cases,'fresh_request':fresh,'tail_accounting':md,'target_dispatch_contract':'T4 full verify plus T1/T2/T3 tails reuse attempt036-qualified DS41 rowwise target math; no target math change'}
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
