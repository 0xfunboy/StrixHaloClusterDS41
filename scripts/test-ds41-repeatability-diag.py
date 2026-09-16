#!/usr/bin/env python3
from __future__ import annotations
import os,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def batch(req,before,sched,plen=1588,prefill=True):
    positions=torch.arange(before,before+sched,dtype=torch.int64)
    return SimpleNamespace(req_ids=[req],num_computed_prefill_tokens_np=np.array([before]),prefill_len_np=np.array([plen]),
        is_prefilling_np=np.array([prefill]),num_scheduled_tokens=np.array([sched]),num_draft_tokens=0,idx_mapping_np=np.array([0]),
        logits_indices=torch.tensor([sched-1],dtype=torch.int64),positions=positions)
def sampler(tok=7):
    return SimpleNamespace(sampled_token_ids=torch.tensor([[tok]],dtype=torch.int64),num_sampled=torch.tensor([1],dtype=torch.int32),num_rejected=torch.tensor([0],dtype=torch.int32))

def attn_packet(cap,positions,seed):
    g=torch.Generator().manual_seed(seed); n=positions.numel()
    q=torch.randn(n,2,4,generator=g); workspace=torch.randn(1,20,4,generator=g)
    inds=torch.full((n,4),-1,dtype=torch.int32); inds[:,-4:]=torch.tensor([0,2,5,-1],dtype=torch.int32)
    lens=torch.full((n,),3,dtype=torch.int32); topk=torch.arange(n*4,dtype=torch.int32).view(n,4)%10
    cap.record_layer2_attention_prefill(q=q,positions=positions,workspace=workspace,combined_indices=inds,combined_lens=lens,
        topk_indices=topk,seq_lens=torch.tensor([int(positions[-1])+1]),gather_lens=torch.tensor([min(n,128)]),
        query_start_loc=torch.tensor([0,n]),block_table=torch.tensor([[1,2,3]],dtype=torch.int32),
        swa_block_table=torch.tensor([[4,5,6]],dtype=torch.int32),N=8,M=20,scale=.125,attn_sink=torch.zeros(2))
    cap.record_layer2_attention_kernel_output(torch.randn(n,2,4,generator=g))

def record_layer2(cap,n,seed):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(n,4,generator=g); residual=torch.randn(n,2,4,generator=g); mix=torch.randn(n,2,generator=g)
    cap.record_layer2_boundary('entry',x=x,residual=residual,pre_mix=mix,post_mix=mix,res_mix=mix)
    cap.record_layer2_boundary('attn_mhc_pre',x=x,residual=residual,pre_mix=mix,post_mix=mix,res_mix=mix)
    cap.record_layer2_boundary('attn_norm',x=x)
    cap.record_layer2_projection(torch.arange(n),torch.randn(n,4,generator=g))
    cap.record_layer2_boundary('attention_out',x=x)
    cap.record_layer2_boundary('attn_mhc_post',residual=residual)
    cap.record_layer2_boundary('ffn_mhc_pre',x=x,residual=residual,pre_mix=mix,post_mix=mix,res_mix=mix)
    cap.record_layer2_boundary('ffn_norm',x=x)
    cap.record_layer2_boundary('ffn_out',x=x)

def complete(cap,req,offset=0):
    b0=batch(req,0,1023); cap.arm_from_input_batch(b0); assert cap.current is not None
    cap.record_model_input(torch.empty(0,dtype=torch.int32),torch.empty(0,dtype=torch.int64),torch.empty((0,4)))
    x0=torch.randn(1023,4); cap.record_model_input(torch.arange(1023,dtype=torch.int32)+offset,torch.arange(1023),x0)
    cap.record_layer(0,torch.randn(1023,4),torch.randn(1023,2,4),torch.randn(1023,2),torch.randn(1023,2),torch.randn(1023,2))
    cap.record_layer(1,torch.randn(1023,4),torch.randn(1023,2,4),torch.randn(1023,2),torch.randn(1023,2),torch.randn(1023,2))
    record_layer2(cap,1023,100+offset); attn_packet(cap,torch.arange(1023),200+offset)
    assert not cap.current['layers']
    b1=batch(req,1023,565); cap.record_model_input(torch.arange(565,dtype=torch.int32)+10000+offset,torch.arange(1023,1588),torch.randn(565,4))
    cap.record_layer(0,torch.randn(565,4),torch.randn(565,2,4),torch.randn(565,2),torch.randn(565,2),torch.randn(565,2))
    cap.record_layer(1,torch.randn(565,4),torch.randn(565,2,4),torch.randn(565,2),torch.randn(565,2),torch.randn(565,2))
    record_layer2(cap,565,300+offset); attn_packet(cap,torch.arange(1023,1588),400+offset)
    cap.record_layer(39,torch.randn(565,2,4),torch.randn(565,2,4),torch.randn(565,2),torch.randn(565,2),torch.randn(565,2))
    cap.record_final_hidden(torch.randn(565,4)); cap.record_raw_logits(b1,torch.randn(1,16),np.array([False])); cap.finish_after_sample(b1,sampler())

with tempfile.TemporaryDirectory() as td:
    os.environ['DS41_REPEAT_DIAG_DIR']=td; os.environ['DS41_REPEAT_DIAG_MAX_REQUESTS']='2'; os.environ['RANK']='0'
    import runtime.ds41.repeatability_diag as mod
    mod._CAPTURE=None; cap=mod.get_repeatability_capture()
    cap.arm_from_input_batch(batch('dummy',0,10,plen=10)); assert cap.current is None
    complete(cap,'real0'); complete(cap,'real1',1)
    cap.arm_from_input_batch(batch('real2',0,1023)); assert cap.current is None
    for i in (0,1):
        j=json.load(open(Path(td)/f'repeat-rank0-request{i}.json')); assert j['status']=='COMPLETE'; assert j['chunk_ranges']==[[0,1022],[1023,1587]]; assert j['has_raw_logits'] is True
        assert j['layer2_full_chunks']==['0','1']; assert j['layer2_attention_chunks']==['0','1']
        d=torch.load(Path(td)/f'repeat-rank0-request{i}.pt',weights_only=False); assert d['raw_logits']['needs_logits_processing'] is False; assert d['sample']['sampled_token_ids'].item()==7
        for ck in ('0','1'):
            assert d['layer2_attention'][ck]['context_rows'].shape[0]==3
            assert d['layer2_projection'][ck]['kv_current_chunk'].shape[0] in (1023,565)
            assert set(d['early_full'][ck])=={0,1}
    os.environ['DS41_REPEAT_DIAG_DIR']=td+'/invalid'; os.environ['DS41_REPEAT_DIAG_MAX_REQUESTS']='1'; mod._CAPTURE=None; bad=mod.get_repeatability_capture(); bad.arm_from_input_batch(batch('bad',0,1023)); assert bad.current is not None; bad.cleanup_finished(['bad']);
    j=json.load(open(Path(td)/'invalid/repeat-rank0-request0.json')); assert j['status']=='INVALID'
print('REPEATABILITY_DIAG_EXTENDED_MODEL_FREE=PASS')
