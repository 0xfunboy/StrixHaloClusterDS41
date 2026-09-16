"""Bounded DS41 decode/verifier repeatability trace.

Opt-in only. Captures two selected requests through a bounded emitted-token prefix.
It records target verification inputs/logits, sampler decisions, and sparse-indexer
state at index-source layers. GPU tensors are cloned on-stream and materialized once
when the emitted-token cap is reached; no per-step synchronization is introduced.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

class DecodeRepeatTrace:
    def __init__(self) -> None:
        root=os.environ.get('DS41_DECODE_TRACE_DIR','').strip()
        self.enabled=bool(root); self.root=Path(root) if root else None
        self.rank=int(os.environ.get('RANK',os.environ.get('LOCAL_RANK','0')))
        self.prompt_tokens=int(os.environ.get('DS41_DECODE_TRACE_PROMPT_TOKENS','1588'))
        self.max_requests=int(os.environ.get('DS41_DECODE_TRACE_MAX_REQUESTS','2'))
        self.emit_cap=int(os.environ.get('DS41_DECODE_TRACE_EMIT_CAP','40'))
        self.completed=0; self.current:dict[str,Any]|None=None

    @staticmethod
    def _clone(x:Any)->Any:
        if x is None: return None
        try: return x.detach().clone()
        except Exception: return x

    def _req_index(self,input_batch:Any)->int|None:
        if self.current is None: return None
        rid=self.current['request_id']
        for i,x in enumerate(input_batch.req_ids):
            if str(x)==rid: return i
        return None

    def begin_step(self,input_batch:Any) -> None:
        if not self.enabled or self.completed>=self.max_requests: return
        if self.current is None:
            for i,req_id in enumerate(input_batch.req_ids):
                before=int(input_batch.num_computed_prefill_tokens_np[i]); plen=int(input_batch.prefill_len_np[i])
                if bool(input_batch.is_prefilling_np[i]) and before==0 and plen==self.prompt_tokens:
                    self.current={'schema':'ds41-decode-repeat-trace-v1','rank':self.rank,
                        'request_index':self.completed,'request_id':str(req_id),'prompt_tokens':plen,
                        'emit_cap':self.emit_cap,'emitted_count':0,'emitted_ids':[],'steps':[],
                        'invalid_reason':None}
                    break
        i=self._req_index(input_batch)
        if i is None: return
        s=int(input_batch.cu_num_logits_np[i]); e=int(input_batch.cu_num_logits_np[i+1])
        li=input_batch.logits_indices[s:e]
        draft_per_req=(None if input_batch.num_draft_tokens_per_req is None else int(input_batch.num_draft_tokens_per_req[i]))
        step={'ordinal':len(self.current['steps']),'emitted_prefix_len':int(self.current['emitted_count']),
            'num_computed_tokens':int(input_batch.num_computed_tokens_np[i]),
            'num_computed_prefill_tokens':int(input_batch.num_computed_prefill_tokens_np[i]),
            'prefill_len':int(input_batch.prefill_len_np[i]),'is_prefilling':bool(input_batch.is_prefilling_np[i]),
            'num_scheduled_tokens':int(input_batch.num_scheduled_tokens[i]),'num_draft_tokens_per_req':draft_per_req,
            'logit_slice':[s,e],'target_input_ids':self._clone(input_batch.input_ids[li]),
            'target_positions':self._clone(input_batch.positions[li]),
            'expanded_local_pos':self._clone(input_batch.expanded_local_pos[s:e]),
            'raw_logits':None,'needs_logits_processing':None,'indexer':{},'sampler':None}
        self.current['steps'].append(step)

    def record_raw_logits(self,input_batch:Any,logits:Any,needs_processing:Any) -> None:
        i=self._req_index(input_batch)
        if i is None or self.current is None or not self.current['steps']: return
        step=self.current['steps'][-1]
        s,e=step['logit_slice']
        step['raw_logits']=self._clone(logits[s:e])
        try: step['needs_logits_processing']=bool(needs_processing[i])
        except Exception: step['needs_logits_processing']=None

    def record_indexer(self,layer_id:int,positions:Any,index_meta:Any,topk:Any) -> None:
        if self.current is None or not self.current['steps']: return
        nd=int(getattr(index_meta,'num_decode_tokens',0)); npf=int(getattr(index_meta,'num_prefill_tokens',0))
        if nd<=0: return
        # Diagnostic request runs alone; fail closed if the decode rows exceed supplied positions.
        n=min(nd,int(positions.shape[0]))
        self.current['steps'][-1]['indexer'][str(int(layer_id))]={
            'num_decode_tokens':nd,'num_prefill_tokens':npf,
            'positions':self._clone(positions[:n]),'topk':self._clone(topk[:n])}

    def record_decode_attention(self,layer_id:int,q:Any,swa_indices:Any,swa_lens:Any,
                                topk_ragged_indices:Any,topk_ragged_indptr:Any,topk_lens:Any) -> None:
        if self.current is None or not self.current['steps']: return
        d=self.current['steps'][-1]['indexer'].get(str(int(layer_id)))
        if d is None: return
        d['q']=self._clone(q); d['swa_indices']=self._clone(swa_indices); d['swa_lens']=self._clone(swa_lens)
        d['topk_ragged_indices']=self._clone(topk_ragged_indices); d['topk_ragged_indptr']=self._clone(topk_ragged_indptr); d['topk_lens']=self._clone(topk_lens)

    def record_decode_attention_output(self,layer_id:int,output:Any) -> None:
        if self.current is None or not self.current['steps']: return
        d=self.current['steps'][-1]['indexer'].get(str(int(layer_id)))
        if d is not None: d['attention_output']=self._clone(output)

    def record_sampler(self,input_batch:Any,sampler_output:Any) -> None:
        i=self._req_index(input_batch)
        if i is None or self.current is None or not self.current['steps']: return
        step=self.current['steps'][-1]
        ns=int(sampler_output.num_sampled[i]); nr=int(sampler_output.num_rejected[i])
        row=self._clone(sampler_output.sampled_token_ids[i])
        valid=self._clone(sampler_output.sampled_token_ids[i,:ns])
        step['sampler']={'num_sampled':ns,'num_rejected':nr,'sampled_row':row,'emitted_ids':valid}
        try: ids=[int(x) for x in sampler_output.sampled_token_ids[i,:ns].detach().cpu().tolist()]
        except Exception: ids=[]
        self.current['emitted_ids'].extend(ids); self.current['emitted_count']+=len(ids)
        if self.current['emitted_count']>=self.emit_cap: self._finalize(False)

    def cleanup_finished(self,finished_req_ids:Any) -> None:
        if self.current is None: return
        if self.current['request_id'] in {str(x) for x in finished_req_ids}:
            self._finalize(False)

    def _cpu(self,x:Any)->Any:
        if isinstance(x,dict): return {k:self._cpu(v) for k,v in x.items()}
        if isinstance(x,list): return [self._cpu(v) for v in x]
        try:
            import torch
            if isinstance(x,torch.Tensor): return x.detach().cpu()
        except Exception: pass
        return x

    def _finalize(self,invalid:bool) -> None:
        cur=self.current
        if cur is None: return
        try:
            assert self.root is not None; self.root.mkdir(parents=True,exist_ok=True)
            cpu=self._cpu(cur)
            import torch
            base=self.root/f'decode-trace-rank{self.rank}-request{self.completed}'
            torch.save(cpu,base.with_suffix('.pt'))
            summary={'schema':cpu['schema'],'rank':self.rank,'request_index':self.completed,
                'request_id':cpu['request_id'],'status':'INVALID' if invalid else 'COMPLETE',
                'step_count':len(cpu['steps']),'emitted_count':cpu['emitted_count'],
                'emitted_ids':cpu['emitted_ids'],'invalid_reason':cpu.get('invalid_reason')}
            base.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
        finally:
            self.completed+=1; self.current=None

_TRACE:DecodeRepeatTrace|None=None
def get_decode_repeat_trace()->DecodeRepeatTrace:
    global _TRACE
    if _TRACE is None: _TRACE=DecodeRepeatTrace()
    return _TRACE
