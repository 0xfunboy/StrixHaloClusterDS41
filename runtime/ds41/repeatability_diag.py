"""Bounded DS41 same-arm repeatability capture for V2 serving.

Opt-in via DS41_REPEAT_DIAG_DIR. The runner arms only a real selected prefill
request. Target model hooks clone GPU tensors without CPU synchronization; one
materialization happens after the first-token sample of the final prefill chunk.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

class RepeatabilityCapture:
    def __init__(self) -> None:
        root=os.environ.get('DS41_REPEAT_DIAG_DIR','').strip()
        self.enabled=bool(root); self.root=Path(root) if root else None
        self.rank=int(os.environ.get('RANK',os.environ.get('LOCAL_RANK','0')))
        self.prompt_tokens=int(os.environ.get('DS41_REPEAT_DIAG_PROMPT_TOKENS','1588'))
        self.max_requests=int(os.environ.get('DS41_REPEAT_DIAG_MAX_REQUESTS','2'))
        self.completed=0; self.current:dict[str,Any]|None=None

    def arm_from_input_batch(self,input_batch:Any) -> None:
        if not self.enabled or self.current is not None or self.completed>=self.max_requests: return
        for i,req_id in enumerate(input_batch.req_ids):
            before=int(input_batch.num_computed_prefill_tokens_np[i]); plen=int(input_batch.prefill_len_np[i])
            if bool(input_batch.is_prefilling_np[i]) and before==0 and plen==self.prompt_tokens:
                self.current={'schema':'ds41-repeat-diag-v1','rank':self.rank,'request_index':self.completed,
                              'request_id':str(req_id),'prompt_tokens':plen,'chunks':[],'layers':{},
                              'final_hidden':None,'raw_logits':None,'sample':None,'invalid_reason':None}
                return

    def cleanup_finished(self,finished_req_ids:Any) -> None:
        if self.current is None: return
        if self.current['request_id'] in {str(x) for x in finished_req_ids}:
            self.current['invalid_reason']='request finished before diagnostic finalization'
            self._finalize(invalid=True)

    def _for_current(self) -> bool: return self.enabled and self.current is not None

    def record_model_input(self,input_ids:Any,positions:Any,hidden_states:Any) -> None:
        if not self._for_current() or positions is None or positions.numel()==0: return
        lo=int(positions[0]); hi=int(positions[-1]);
        if lo<0 or hi>=self.prompt_tokens: return
        cur=self.current; assert cur is not None
        chunk={'position_start':lo,'position_end':hi,'tokens':int(positions.numel()),
               'positions':positions.detach().clone(),
               'input_ids':None if input_ids is None else input_ids.detach().clone(),
               'model_input_last':hidden_states[-1].detach().clone()}
        cur['chunks'].append(chunk)
        cur['_capture_layers']=(hi==self.prompt_tokens-1)

    def record_layer(self,idx:int,hidden_states:Any,residual:Any,post_mix:Any,res_mix:Any,pre_mix:Any) -> None:
        if not self._for_current(): return
        cur=self.current; assert cur is not None
        if not cur.get('_capture_layers'): return
        def last(x:Any): return None if x is None else x[-1].detach().clone()
        cur['layers'][int(idx)]={'hidden_states':last(hidden_states),'residual':last(residual),
                                 'post_mix':last(post_mix),'res_mix':last(res_mix),'pre_mix':last(pre_mix)}

    def record_final_hidden(self,hidden_states:Any) -> None:
        if not self._for_current(): return
        cur=self.current; assert cur is not None
        if cur.get('_capture_layers'): cur['final_hidden']=hidden_states[-1].detach().clone()

    def record_raw_logits(self,input_batch:Any,logits:Any,needs_processing:Any) -> None:
        if not self._for_current(): return
        cur=self.current; assert cur is not None
        for i,req_id in enumerate(input_batch.req_ids):
            if str(req_id)!=cur['request_id']: continue
            before=int(input_batch.num_computed_prefill_tokens_np[i]); plen=int(input_batch.prefill_len_np[i]); sched=int(input_batch.num_scheduled_tokens[i])
            if before<plen and before+sched>=plen:
                li=input_batch.logits_indices.detach().clone() if hasattr(input_batch,'logits_indices') else None
                lp=(input_batch.positions[input_batch.logits_indices].detach().clone()
                    if hasattr(input_batch,'logits_indices') and hasattr(input_batch,'positions') else None)
                cur['raw_logits']={'shape':list(logits.shape),'logits':logits.detach().clone(),
                                   'num_draft_tokens':int(input_batch.num_draft_tokens),
                                   'needs_logits_processing':bool(needs_processing[i]),
                                   'logits_indices':li,'logit_positions':lp}
            return

    def finish_after_sample(self,input_batch:Any,sampler_output:Any) -> None:
        if not self._for_current(): return
        cur=self.current; assert cur is not None
        for i,req_id in enumerate(input_batch.req_ids):
            if str(req_id)!=cur['request_id']: continue
            before=int(input_batch.num_computed_prefill_tokens_np[i]); plen=int(input_batch.prefill_len_np[i]); sched=int(input_batch.num_scheduled_tokens[i])
            if before<plen and before+sched>=plen:
                cur['sample']={'sampled_token_ids':sampler_output.sampled_token_ids.detach().clone(),
                               'num_sampled':sampler_output.num_sampled.detach().clone(),
                               'num_rejected':sampler_output.num_rejected.detach().clone()}
                self._finalize(invalid=False)
            return

    def _cpu(self,x:Any)->Any:
        if isinstance(x,dict): return {k:self._cpu(v) for k,v in x.items() if not str(k).startswith('_')}
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
            path=self.root/f"repeat-rank{self.rank}-request{self.completed}.pt"; torch.save(cpu,path)
            summary={k:cpu.get(k) for k in ('schema','rank','request_index','request_id','prompt_tokens','invalid_reason')}
            summary['status']='INVALID' if invalid or cpu.get('invalid_reason') else 'COMPLETE'
            summary['chunk_ranges']=[ [c['position_start'],c['position_end']] for c in cpu.get('chunks',[]) ]
            summary['layer_count']=len(cpu.get('layers',{})); summary['has_raw_logits']=cpu.get('raw_logits') is not None
            (self.root/f"repeat-rank{self.rank}-request{self.completed}.json").write_text(json.dumps(summary,indent=2)+'\n')
        finally:
            self.completed+=1; self.current=None

_CAPTURE:RepeatabilityCapture|None=None
def get_repeatability_capture()->RepeatabilityCapture:
    global _CAPTURE
    if _CAPTURE is None: _CAPTURE=RepeatabilityCapture()
    return _CAPTURE
