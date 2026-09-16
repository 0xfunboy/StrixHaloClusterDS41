#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch

def metrics(a,b):
    af=a.float(); bf=b.float(); d=af-bf; bn=float(bf.norm())
    return {'exact':bool(torch.equal(a,b)),'max_abs':float(d.abs().max()) if d.numel() else 0.0,
            'mean_abs':float(d.abs().mean()) if d.numel() else 0.0,
            'rel_l2':float(d.norm())/max(bn,1e-30) if d.numel() else 0.0}
def top2(x):
    row=x[0].float() if x.ndim==2 else x.float()
    v,i=torch.topk(row,2)
    return {'top1_id':int(i[0]),'top1_logit':float(v[0]),'top2_id':int(i[1]),'top2_logit':float(v[1]),'margin':float(v[0]-v[1])}
def load(p): return torch.load(p,weights_only=False,map_location='cpu')
def compare(a,b):
    out={'schema':'ds41-repeatability-diag-compare-v1','rank':int(a['rank']),'request_ids':[a['request_id'],b['request_id']]}
    ca,cb=a['chunks'],b['chunks']; out['chunks']={'count_a':len(ca),'count_b':len(cb),'rows':[],'input_exact':True}
    for i in range(max(len(ca),len(cb))):
        if i>=len(ca) or i>=len(cb): out['chunks']['input_exact']=False; continue
        ra,rb=ca[i],cb[i]; row={'i':i,'range_a':[ra['position_start'],ra['position_end']],'range_b':[rb['position_start'],rb['position_end']],
             'positions_exact':torch.equal(ra['positions'],rb['positions']),
             'input_ids_exact':(ra['input_ids'] is None and rb['input_ids'] is None) or (ra['input_ids'] is not None and rb['input_ids'] is not None and torch.equal(ra['input_ids'],rb['input_ids'])),
             'model_input_last':metrics(ra['model_input_last'],rb['model_input_last'])}
        out['chunks']['rows'].append(row); out['chunks']['input_exact'] &= row['positions_exact'] and row['input_ids_exact']
    out['layers']=[]; first=None
    out['layer2_boundaries']=[]
    ba=a.get('layer2_boundaries') or {}; bb=b.get('layer2_boundaries') or {}
    order=['entry','attn_mhc_pre','attn_norm','attention_out','attn_mhc_post','ffn_mhc_pre','ffn_norm','ffn_out']
    for name in order:
        va=ba.get(name); vb=bb.get(name)
        if va is None and vb is None: continue
        row={'boundary':name,'fields':{}}
        for k in sorted(set((va or {}).keys())|set((vb or {}).keys())):
            xa=None if va is None else va.get(k); xb=None if vb is None else vb.get(k)
            if xa is None or xb is None: m={'exact':xa is None and xb is None,'missing':True}
            else: m=metrics(xa,xb)
            row['fields'][k]=m
            if first is None and not m.get('exact',False): first={'boundary':f'layer2.{name}.{k}',**m}
        out['layer2_boundaries'].append(row)
    keys=['hidden_states','residual','post_mix','res_mix','pre_mix']
    for li in sorted(set(a['layers'])|set(b['layers']), key=int):
        la=a['layers'].get(li) or a['layers'].get(int(li)); lb=b['layers'].get(li) or b['layers'].get(int(li)); r={'layer':int(li),'fields':{}}
        for k in keys:
            xa=None if la is None else la.get(k); xb=None if lb is None else lb.get(k)
            if xa is None or xb is None: m={'exact':xa is None and xb is None,'missing':True}
            else: m=metrics(xa,xb)
            r['fields'][k]=m
            if first is None and not m.get('exact',False): first={'boundary':f'layer{li}.{k}',**m}
        out['layers'].append(r)
    out['final_hidden']=metrics(a['final_hidden'],b['final_hidden'])
    ra,rb=a['raw_logits'],b['raw_logits']; out['raw_logits']={'metrics':metrics(ra['logits'],rb['logits']),'top2_a':top2(ra['logits']),'top2_b':top2(rb['logits']),
        'shape_a':ra['shape'],'shape_b':rb['shape'],'needs_processing_a':ra['needs_logits_processing'],'needs_processing_b':rb['needs_logits_processing'],
        'num_draft_tokens_a':ra['num_draft_tokens'],'num_draft_tokens_b':rb['num_draft_tokens'],
        'logit_positions_a':ra.get('logit_positions').tolist() if ra.get('logit_positions') is not None else None,
        'logit_positions_b':rb.get('logit_positions').tolist() if rb.get('logit_positions') is not None else None}
    if first is None and not out['final_hidden']['exact']: first={'boundary':'final_hidden',**out['final_hidden']}
    if first is None and not out['raw_logits']['metrics']['exact']: first={'boundary':'raw_logits',**out['raw_logits']['metrics']}
    out['first_divergence']=first
    out['sample_a']={k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in a['sample'].items()}
    out['sample_b']={k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in b['sample'].items()}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--out',required=True); q=ap.parse_args()
    a,b=load(q.a),load(q.b); out=compare(a,b); Path(q.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'first_divergence':out['first_divergence'],'input_exact':out['chunks']['input_exact'],'raw_logits':out['raw_logits']},indent=2))
if __name__=='__main__': main()
