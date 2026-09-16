#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

def metrics(a,b):
    if a is None or b is None: return {'exact':a is None and b is None,'missing':True}
    if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor): return {'exact':a==b,'value_a':a,'value_b':b}
    if tuple(a.shape)!=tuple(b.shape): return {'exact':False,'shape_a':list(a.shape),'shape_b':list(b.shape)}
    eq=torch.equal(a,b); d=a.float()-b.float(); bn=float(b.float().norm())
    out={'exact':bool(eq),'max_abs':float(d.abs().max()) if d.numel() else 0.0,'mean_abs':float(d.abs().mean()) if d.numel() else 0.0,'rel_l2':float(d.norm())/max(bn,1e-30) if d.numel() else 0.0,'shape':list(a.shape),'dtype':str(a.dtype)}
    if not eq and a.ndim>=1 and a.shape[0]>0:
        neq=(a!=b).reshape(a.shape[0],-1).any(dim=1); nz=torch.nonzero(neq,as_tuple=False)
        if nz.numel(): out['first_diff_row']=int(nz[0])
    return out

def top2(x):
    row=x[0].float() if x.ndim==2 else x.float(); v,i=torch.topk(row,2)
    return {'top1_id':int(i[0]),'top1_logit':float(v[0]),'top2_id':int(i[1]),'top2_logit':float(v[1]),'margin':float(v[0]-v[1])}
def load(p): return torch.load(p,weights_only=False,map_location='cpu')

def add(out, first, boundary, a, b, logical_positions=None):
    m=metrics(a,b); out[boundary]=m
    if first[0] is None and not m.get('exact',False):
        f={'boundary':boundary,**m}
        r=m.get('first_diff_row')
        if r is not None and logical_positions is not None and r < len(logical_positions): f['logical_position']=int(logical_positions[r])
        first[0]=f
    return m

def compare(a,b):
    out={'schema':'ds41-repeatability-diag-compare-v2','rank':int(a['rank']),'request_ids':[a['request_id'],b['request_id']],'execution':{}}
    first=[None]
    ca,cb=a['chunks'],b['chunks']; out['chunks']={'count_a':len(ca),'count_b':len(cb),'rows':[],'input_exact':True}
    for ci in range(max(len(ca),len(cb))):
        if ci>=len(ca) or ci>=len(cb): out['chunks']['input_exact']=False; continue
        xa,xb=ca[ci],cb[ci]; pos=xa['positions'].tolist()
        row={'i':ci,'range_a':[xa['position_start'],xa['position_end']],'range_b':[xb['position_start'],xb['position_end']]}
        row['positions']=add(out['execution'],first,f'chunk{ci}.positions',xa['positions'],xb['positions'],pos)
        row['input_ids']=add(out['execution'],first,f'chunk{ci}.input_ids',xa['input_ids'],xb['input_ids'],pos)
        out['chunks']['input_exact'] &= row['positions']['exact'] and row['input_ids']['exact']; out['chunks']['rows'].append(row)
        # Upstream context controls: outputs of layer0/layer1 over every row.
        ea=(a.get('early_full') or {}).get(str(ci),{}); eb=(b.get('early_full') or {}).get(str(ci),{})
        for li in (0,1): add(out['execution'],first,f'chunk{ci}.layer{li}.hidden_full',ea.get(li) or ea.get(str(li)),eb.get(li) or eb.get(str(li)),pos)
        fa=(a.get('layer2_full') or {}).get(str(ci),{}); fb=(b.get('layer2_full') or {}).get(str(ci),{})
        order=['entry','attn_mhc_pre','attn_norm']
        for name in order:
            va,vb=fa.get(name,{}),fb.get(name,{})
            for k in sorted(set(va)|set(vb)): add(out['execution'],first,f'chunk{ci}.layer2.{name}.{k}',va.get(k),vb.get(k),pos)
        pa=(a.get('layer2_projection') or {}).get(str(ci),{}); pb=(b.get('layer2_projection') or {}).get(str(ci),{})
        add(out['execution'],first,f'chunk{ci}.layer2.kv_current_chunk',pa.get('kv_current_chunk'),pb.get('kv_current_chunk'),pos)
        aa=(a.get('layer2_attention') or {}).get(str(ci),{}); ab=(b.get('layer2_attention') or {}).get(str(ci),{})
        for k in ['q_position','q_final','combined_indices_final','combined_lens_final','valid_context_indices','topk_indices_final','seq_lens','gather_lens','query_start_loc','block_table','swa_block_table','context_rows','attn_sink','kernel_output_final']:
            add(out['execution'],first,f'chunk{ci}.layer2.attention.{k}',aa.get(k),ab.get(k),None)
        for k in ['N','M','scale','workspace_shape']:
            add(out['execution'],first,f'chunk{ci}.layer2.attention.{k}',aa.get(k),ab.get(k),None)
        for name in ['attention_out','attn_mhc_post','ffn_mhc_pre','ffn_norm','ffn_out']:
            va,vb=fa.get(name,{}),fb.get(name,{})
            for k in sorted(set(va)|set(vb)): add(out['execution'],first,f'chunk{ci}.layer2.{name}.{k}',va.get(k),vb.get(k),pos)
    # Preserve final-row all-layer evidence, but it comes after bounded full-context localization.
    out['final_row_layers']=[]
    for li in sorted(set(a.get('layers',{}))|set(b.get('layers',{})),key=int):
        la=a.get('layers',{}).get(li) or a.get('layers',{}).get(int(li)); lb=b.get('layers',{}).get(li) or b.get('layers',{}).get(int(li)); r={'layer':int(li),'fields':{}}
        for k in ['hidden_states','residual','post_mix','res_mix','pre_mix']: r['fields'][k]=metrics(None if la is None else la.get(k),None if lb is None else lb.get(k))
        out['final_row_layers'].append(r)
    out['final_hidden']=metrics(a.get('final_hidden'),b.get('final_hidden'))
    ra,rb=a['raw_logits'],b['raw_logits']; out['raw_logits']={'metrics':metrics(ra['logits'],rb['logits']),'top2_a':top2(ra['logits']),'top2_b':top2(rb['logits']),'shape_a':ra['shape'],'shape_b':rb['shape'],'needs_processing_a':ra['needs_logits_processing'],'needs_processing_b':rb['needs_logits_processing'],'num_draft_tokens_a':ra['num_draft_tokens'],'num_draft_tokens_b':rb['num_draft_tokens'],'logit_positions_a':ra.get('logit_positions').tolist() if ra.get('logit_positions') is not None else None,'logit_positions_b':rb.get('logit_positions').tolist() if rb.get('logit_positions') is not None else None}
    if first[0] is None and not out['final_hidden']['exact']: first[0]={'boundary':'final_hidden',**out['final_hidden']}
    if first[0] is None and not out['raw_logits']['metrics']['exact']: first[0]={'boundary':'raw_logits',**out['raw_logits']['metrics']}
    out['first_divergence']=first[0]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--out',required=True); q=ap.parse_args()
    a,b=load(q.a),load(q.b); out=compare(a,b); Path(q.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'first_divergence':out['first_divergence'],'input_exact':out['chunks']['input_exact'],'raw_logits':out['raw_logits']},indent=2))
if __name__=='__main__': main()
