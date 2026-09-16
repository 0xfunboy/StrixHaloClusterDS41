#!/usr/bin/env python3
from __future__ import annotations
import os,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def batch(req,before,sched,plen=1588,prefill=True):
    return SimpleNamespace(req_ids=[req],num_computed_prefill_tokens_np=np.array([before]),prefill_len_np=np.array([plen]),
        is_prefilling_np=np.array([prefill]),num_scheduled_tokens=np.array([sched]),num_draft_tokens=0,idx_mapping_np=np.array([0]))
def sampler(tok=7):
    return SimpleNamespace(sampled_token_ids=torch.tensor([[tok]],dtype=torch.int64),num_sampled=torch.tensor([1],dtype=torch.int32),num_rejected=torch.tensor([0],dtype=torch.int32))

def complete(cap,req,offset=0):
    b0=batch(req,0,1023); cap.arm_from_input_batch(b0); assert cap.current is not None
    # Empty / invalid model callback is inert.
    cap.record_model_input(torch.empty(0,dtype=torch.int32),torch.empty(0,dtype=torch.int64),torch.empty((0,4)))
    cap.record_model_input(torch.arange(1023,dtype=torch.int32)+offset,torch.arange(1023),torch.randn(1023,4))
    cap.record_layer(0,torch.randn(1023,2,4),torch.randn(1023,2,4),torch.randn(1023,2),torch.randn(1023,2),torch.randn(1023,2))
    assert not cap.current['layers']
    b1=batch(req,1023,565); cap.record_model_input(torch.arange(565,dtype=torch.int32)+10000+offset,torch.arange(1023,1588),torch.randn(565,4))
    cap.record_layer(0,torch.randn(565,2,4),torch.randn(565,2,4),torch.randn(565,2),torch.randn(565,2),torch.randn(565,2))
    cap.record_layer(39,torch.randn(565,2,4),torch.randn(565,2,4),torch.randn(565,2),torch.randn(565,2),torch.randn(565,2))
    cap.record_final_hidden(torch.randn(565,4)); cap.record_raw_logits(b1,torch.randn(1,16),np.array([False])); cap.finish_after_sample(b1,sampler())

with tempfile.TemporaryDirectory() as td:
    os.environ['DS41_REPEAT_DIAG_DIR']=td; os.environ['DS41_REPEAT_DIAG_MAX_REQUESTS']='2'; os.environ['RANK']='0'
    import runtime.ds41.repeatability_diag as mod
    mod._CAPTURE=None; cap=mod.get_repeatability_capture()
    # Dummy / wrong prompt never arms.
    cap.arm_from_input_batch(batch('dummy',0,10,plen=10)); assert cap.current is None
    complete(cap,'real0'); complete(cap,'real1',1)
    # Max reached: third request ignored.
    cap.arm_from_input_batch(batch('real2',0,1023)); assert cap.current is None
    for i in (0,1):
        j=json.load(open(Path(td)/f'repeat-rank0-request{i}.json')); assert j['status']=='COMPLETE'; assert j['chunk_ranges']==[[0,1022],[1023,1587]]; assert j['layer_count']==2; assert j['has_raw_logits'] is True
        d=torch.load(Path(td)/f'repeat-rank0-request{i}.pt',weights_only=False); assert d['raw_logits']['needs_logits_processing'] is False; assert d['sample']['sampled_token_ids'].item()==7
    # Separate invalid cleanup path.
    os.environ['DS41_REPEAT_DIAG_DIR']=td+'/invalid'; os.environ['DS41_REPEAT_DIAG_MAX_REQUESTS']='1'; mod._CAPTURE=None; bad=mod.get_repeatability_capture(); bad.arm_from_input_batch(batch('bad',0,1023)); assert bad.current is not None; bad.cleanup_finished(['bad']);
    j=json.load(open(Path(td)/'invalid/repeat-rank0-request0.json')); assert j['status']=='INVALID'
print('REPEATABILITY_DIAG_MODEL_FREE=PASS')
