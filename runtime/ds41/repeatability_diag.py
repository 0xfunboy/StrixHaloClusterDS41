"""Bounded DS41 same-arm repeatability capture for V2 serving.

Opt-in via DS41_REPEAT_DIAG_DIR. The runner arms only a real selected prefill
request. All tensors are cloned on the current GPU stream; no per-layer CPU
synchronization is introduced. One materialization happens after the first-token
sample of the final prefill chunk.

The layer2 diagnostic stores both engine chunks. It keeps the full logical inputs
needed to distinguish an upstream context difference from attention variability,
and for the observed final query it stores the exact gathered KV rows selected by
the ROCm sparse prefill backend rather than dumping the whole physical cache.
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
        self.mode=os.environ.get('DS41_REPEAT_DIAG_MODE','full').strip() or 'full'
        if self.mode not in ('full','raw-only'): raise ValueError(f'invalid DS41_REPEAT_DIAG_MODE={self.mode}')
        self.completed=0; self.current:dict[str,Any]|None=None

    def arm_from_input_batch(self,input_batch:Any) -> None:
        if not self.enabled or self.current is not None or self.completed>=self.max_requests: return
        for i,req_id in enumerate(input_batch.req_ids):
            before=int(input_batch.num_computed_prefill_tokens_np[i]); plen=int(input_batch.prefill_len_np[i])
            if bool(input_batch.is_prefilling_np[i]) and before==0 and plen==self.prompt_tokens:
                self.current={'schema':'ds41-repeat-diag-v2','rank':self.rank,'request_index':self.completed,
                              'request_id':str(req_id),'prompt_tokens':plen,'chunks':[],'layers':{},
                              'early_full':{},'layer2_boundaries':{},'layer2_final_rows':{},
                              'layer2_full':{},'layer2_projection':{},'layer2_attention':{},
                              'final_hidden':None,'raw_logits':None,'sample':None,'invalid_reason':None,'mode':self.mode,'canonical_topk_calls':[]}
                return

    def cleanup_finished(self,finished_req_ids:Any) -> None:
        if self.current is None: return
        if self.current['request_id'] in {str(x) for x in finished_req_ids}:
            self.current['invalid_reason']='request finished before diagnostic finalization'
            self._finalize(invalid=True)

    def _for_current(self) -> bool: return self.enabled and self.current is not None
    def _chunk_index(self) -> int:
        cur=self.current; assert cur is not None
        return max(len(cur.get('chunks',[]))-1,0)
    @staticmethod
    def _clone(x:Any)->Any:
        if x is None: return None
        try: return x.detach().clone()
        except Exception: return x

    def record_model_input(self,input_ids:Any,positions:Any,hidden_states:Any) -> None:
        if not self._for_current() or positions is None or positions.numel()==0: return
        lo=int(positions[0]); hi=int(positions[-1])
        if lo<0 or hi>=self.prompt_tokens: return
        cur=self.current; assert cur is not None
        chunk={'position_start':lo,'position_end':hi,'tokens':int(positions.numel()),
               'positions':positions.detach().clone(),
               'input_ids':None if input_ids is None else input_ids.detach().clone(),
               'model_input_last':hidden_states[-1].detach().clone()}
        cur['chunks'].append(chunk)
        cur['_capture_layers']=(hi==self.prompt_tokens-1)

    def record_canonical_topk(self,layer_id:int,rows:int,width:int,num_decode:int,num_prefill:int) -> None:
        if not self._for_current(): return
        cur=self.current; assert cur is not None
        cur.setdefault('canonical_topk_calls',[]).append({
            'layer_id':int(layer_id),'rows':int(rows),'width':int(width),
            'num_decode_tokens':int(num_decode),'num_prefill_tokens':int(num_prefill),
        })

    def record_layer2_boundary(self,name:str,**values:Any) -> None:
        if not self._for_current(): return
        if self.mode=='raw-only': return
        cur=self.current; assert cur is not None
        ci=self._chunk_index(); ck=str(ci); name=str(name)
        # Always keep a stable final-row indicator per chunk.
        finals=cur.setdefault('layer2_final_rows',{}).setdefault(ck,{})
        finals[name]={k:(None if v is None else self._clone(v[-1])) for k,v in values.items()}
        if cur.get('_capture_layers'):
            cur['layer2_boundaries'][name]=finals[name]
        # Full tensors only for the bounded replay/localization boundaries.
        keep={
            'entry':{'x','residual','pre_mix','post_mix','res_mix'},
            'attn_mhc_pre':{'x','residual','pre_mix','post_mix','res_mix'},
            'attn_norm':{'x'},
            'attention_out':{'x'},
            'attn_mhc_post':{'residual'},
            'ffn_mhc_pre':{'x','residual','pre_mix','post_mix','res_mix'},
            'ffn_norm':{'x'},
            'ffn_out':{'x'},
        }.get(name,set())
        if keep:
            dst=cur.setdefault('layer2_full',{}).setdefault(ck,{}).setdefault(name,{})
            for k,v in values.items():
                if k in keep: dst[k]=self._clone(v)

    def record_layer2_projection(self,positions:Any,kv:Any) -> None:
        if not self._for_current() or positions is None or positions.numel()==0: return
        if self.mode=='raw-only': return
        cur=self.current; assert cur is not None
        ci=self._chunk_index(); ck=str(ci)
        cur.setdefault('layer2_projection',{})[ck]={
            'position_start':int(positions[0]),'position_end':int(positions[-1]),
            'positions':self._clone(positions),'kv_current_chunk':self._clone(kv),
        }

    def record_layer2_attention_prefill(self,*,q:Any,positions:Any,workspace:Any,
                                        combined_indices:Any,combined_lens:Any,
                                        topk_indices:Any,seq_lens:Any,gather_lens:Any,
                                        query_start_loc:Any,block_table:Any,swa_block_table:Any,
                                        N:int,M:int,scale:float,attn_sink:Any) -> None:
        if not self._for_current() or q is None or q.shape[0]==0: return
        if self.mode=='raw-only': return
        cur=self.current; assert cur is not None
        ci=self._chunk_index(); ck=str(ci)
        row=combined_indices[-1]
        # Build the exact valid prefix on GPU; no .item()/CPU sync here.
        import torch
        cols=torch.arange(row.numel(),device=row.device)
        mask=(cols < combined_lens[-1]) & (row >= 0)
        valid_idx=row[mask].to(torch.long)
        flat=workspace.view(-1,q.shape[-1])
        context_rows=flat.index_select(0,valid_idx)
        cur.setdefault('layer2_attention',{})[ck]={
            'position_start':int(positions[0]),'position_end':int(positions[-1]),
            'q_position':self._clone(positions[-1:]),
            'q_final':self._clone(q[-1]),
            'combined_indices_final':self._clone(row),
            'combined_lens_final':self._clone(combined_lens[-1:]),
            'valid_context_indices':self._clone(valid_idx),
            'context_rows':self._clone(context_rows),
            'topk_indices_final':self._clone(topk_indices[-1]),
            'seq_lens':self._clone(seq_lens),'gather_lens':self._clone(gather_lens),
            'query_start_loc':self._clone(query_start_loc),
            'block_table':self._clone(block_table),'swa_block_table':self._clone(swa_block_table),
            'workspace_shape':list(workspace.shape),'N':int(N),'M':int(M),
            'scale':float(scale),'attn_sink':self._clone(attn_sink),
            'kernel_output_final':None,
        }

    def record_layer2_attention_kernel_output(self,output:Any) -> None:
        if not self._for_current() or output is None or output.shape[0]==0: return
        if self.mode=='raw-only': return
        cur=self.current; assert cur is not None
        ck=str(self._chunk_index()); pkt=cur.setdefault('layer2_attention',{}).get(ck)
        if pkt is not None: pkt['kernel_output_final']=self._clone(output[-1])

    def record_layer(self,idx:int,hidden_states:Any,residual:Any,post_mix:Any,res_mix:Any,pre_mix:Any) -> None:
        if not self._for_current(): return
        if self.mode=='raw-only': return
        cur=self.current; assert cur is not None
        ci=self._chunk_index(); ck=str(ci)
        # Full early-layer output is a bounded upstream control for contextual
        # differences that final-row-only capture could miss.
        if int(idx) in (0,1):
            cur.setdefault('early_full',{}).setdefault(ck,{})[int(idx)]=self._clone(hidden_states)
        if not cur.get('_capture_layers'): return
        def last(x:Any): return None if x is None else self._clone(x[-1])
        cur['layers'][int(idx)]={'hidden_states':last(hidden_states),'residual':last(residual),
                                 'post_mix':last(post_mix),'res_mix':last(res_mix),'pre_mix':last(pre_mix)}

    def record_final_hidden(self,hidden_states:Any) -> None:
        if not self._for_current(): return
        if self.mode=='raw-only': return
        cur=self.current; assert cur is not None
        if cur.get('_capture_layers'): cur['final_hidden']=self._clone(hidden_states[-1])

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
                cur['raw_logits']={'shape':list(logits.shape),'logits':self._clone(logits),
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
                # Full mode requires both layer2 engine chunks; raw-only intentionally does not.
                if self.mode=='full':
                    for key in ('layer2_full','layer2_projection','layer2_attention'):
                        got=cur.get(key,{})
                        if set(got.keys())!={'0','1'}:
                            cur['invalid_reason']=f'missing {key} chunks: {sorted(got.keys())}'
                cur['sample']={'sampled_token_ids':self._clone(sampler_output.sampled_token_ids),
                               'num_sampled':self._clone(sampler_output.num_sampled),
                               'num_rejected':self._clone(sampler_output.num_rejected)}
                self._finalize(invalid=bool(cur.get('invalid_reason')))
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
            summary['layer2_full_chunks']=sorted(cpu.get('layer2_full',{}).keys())
            summary['layer2_attention_chunks']=sorted(cpu.get('layer2_attention',{}).keys())
            (self.root/f"repeat-rank{self.rank}-request{self.completed}.json").write_text(json.dumps(summary,indent=2)+'\n')
        finally:
            self.completed+=1; self.current=None

_CAPTURE:RepeatabilityCapture|None=None
def get_repeatability_capture()->RepeatabilityCapture:
    global _CAPTURE
    if _CAPTURE is None: _CAPTURE=RepeatabilityCapture()
    return _CAPTURE
