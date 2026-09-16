#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

def met(a,b):
    if a is None or b is None: return {'exact':a is None and b is None,'missing':True}
    exact=bool(torch.equal(a,b)); out={'exact':exact,'shape_a':list(a.shape),'shape_b':list(b.shape),'dtype':str(a.dtype)}
    if a.shape==b.shape and a.numel() and a.dtype.is_floating_point:
        d=a.float()-b.float(); out.update(max_abs=float(d.abs().max()),rel_l2=float(d.norm())/max(float(b.float().norm()),1e-30))
    return out

def topk_rows(a,b):
    out=[]
    n=min(a.shape[0],b.shape[0])
    for r in range(n):
        xa=a[r][a[r]>=0].to(torch.int64); xb=b[r][b[r]>=0].to(torch.int64)
        sa=xa.sort().values; sb=xb.sort().values
        out.append({'row':r,'exact_order':bool(torch.equal(xa,xb)),'same_multiset':bool(torch.equal(sa,sb)),
                    'count_a':int(xa.numel()),'count_b':int(xb.numel())})
    return out

def valid_ids(step):
    s=step.get('sampler') or {}; ids=s.get('emitted_ids')
    return [] if ids is None else [int(x) for x in ids.tolist()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--out',required=True); q=ap.parse_args()
    A=torch.load(q.a,weights_only=False,map_location='cpu'); B=torch.load(q.b,weights_only=False,map_location='cpu')
    pa=[]; pb=[]; rows=[]; first=None
    n=min(len(A['steps']),len(B['steps']))
    for k in range(n):
        a=A['steps'][k]; b=B['steps'][k]
        cond={'prefix_exact':pa==pb,'prefix_len_a':len(pa),'prefix_len_b':len(pb),
              'positions':met(a['target_positions'],b['target_positions']),
              'target_input_ids':met(a['target_input_ids'],b['target_input_ids']),
              'expanded_local_pos':met(a['expanded_local_pos'],b['expanded_local_pos']),
              'num_draft_tokens_per_req':[a['num_draft_tokens_per_req'],b['num_draft_tokens_per_req']]}
        comparable=cond['prefix_exact'] and cond['positions']['exact'] and cond['target_input_ids']['exact'] and cond['expanded_local_pos']['exact']
        rr={'step':k,'comparable_conditioning':comparable,'conditioning':cond,'indexer':{},'raw_logits':None,'sampler':{}}
        if comparable:
            for li in sorted(set(a.get('indexer',{}))|set(b.get('indexer',{})),key=int):
                ia=a.get('indexer',{}).get(li); ib=b.get('indexer',{}).get(li)
                if ia is None or ib is None:
                    rr['indexer'][li]={'missing':True}; continue
                tr=topk_rows(ia['topk'],ib['topk'])
                rr['indexer'][li]={'topk_rows':tr,'q':met(ia.get('q'),ib.get('q')),
                    'swa_indices_raw':met(ia.get('swa_indices'),ib.get('swa_indices')),
                    'swa_lens':met(ia.get('swa_lens'),ib.get('swa_lens')),
                    'attention_output':met(ia.get('attention_output'),ib.get('attention_output'))}
                if first is None:
                    bad=[x for x in tr if not x['exact_order']]
                    if bad: first={'step':k,'boundary':f'layer{li}.topk_order','detail':bad[0]}
                    elif not rr['indexer'][li]['q'].get('exact',False): first={'step':k,'boundary':f'layer{li}.q','detail':rr['indexer'][li]['q']}
                    elif not rr['indexer'][li]['attention_output'].get('exact',False): first={'step':k,'boundary':f'layer{li}.attention_output','detail':rr['indexer'][li]['attention_output']}
            rr['raw_logits']=met(a.get('raw_logits'),b.get('raw_logits'))
            if first is None and not rr['raw_logits'].get('exact',False): first={'step':k,'boundary':'raw_target_logits','detail':rr['raw_logits']}
        elif first is None:
            first={'step':k,'boundary':'conditioning_input_or_step_boundary','detail':cond}
        sa=a.get('sampler') or {}; sb=b.get('sampler') or {}
        rr['sampler']={'num_sampled':[sa.get('num_sampled'),sb.get('num_sampled')],
            'num_rejected':[sa.get('num_rejected'),sb.get('num_rejected')],
            'emitted_ids_a':valid_ids(a),'emitted_ids_b':valid_ids(b)}
        if first is None and (rr['sampler']['num_sampled'][0]!=rr['sampler']['num_sampled'][1] or rr['sampler']['num_rejected'][0]!=rr['sampler']['num_rejected'][1] or rr['sampler']['emitted_ids_a']!=rr['sampler']['emitted_ids_b']):
            first={'step':k,'boundary':'sampler_acceptance_or_emission','detail':rr['sampler']}
        rows.append(rr); pa.extend(valid_ids(a)); pb.extend(valid_ids(b))
        # Once emitted prefixes diverge, later target inputs are not comparable.
        if pa!=pb: break
    common=0
    ea=[int(x) for x in A.get('emitted_ids',[])]; eb=[int(x) for x in B.get('emitted_ids',[])]
    for x,y in zip(ea,eb):
        if x!=y: break
        common+=1
    out={'schema':'ds41-decode-repeat-compare-v1','rank':A['rank'],'request_ids':[A['request_id'],B['request_id']],
         'steps_compared':len(rows),'first_state_divergence':first,'emitted_common_prefix':common,
         'emitted_first_diff':[ea[common] if common<len(ea) else None,eb[common] if common<len(eb) else None], 'rows':rows}
    Path(q.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'first_state_divergence':first,'emitted_common_prefix':common,'emitted_first_diff':out['emitted_first_diff'],'steps_compared':len(rows)},indent=2))
if __name__=='__main__': main()
