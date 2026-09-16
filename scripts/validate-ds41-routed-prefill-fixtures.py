#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import torch

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(vals,p):
    if not vals:return None
    s=sorted(vals); return s[min(len(s)-1,max(0,math.ceil(p*len(s))-1))]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dir',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    root=Path(a.dir); rows=[]; expected={0,20,40,60}
    for rank in (0,1):
        st=root/f'route-fixture-rank{rank}-status.txt'
        assert st.read_text().strip()=='COMPLETE', (rank,st.read_text())
        seen=set()
        for p in sorted(root.glob(f'route-fixture-rank{rank}-call*.pt')):
            d=torch.load(p,weights_only=False); idx=int(d['call_index']); seen.add(idx)
            assert d['schema']=='ds41-routed-prefill-fixture-v2'
            assert int(d['hidden'])==5120 and int(d['top_k'])==6 and int(d['w13_quant'])==16 and int(d['w2_quant'])==10
            ids=d['topk_ids'].to(torch.int64); em=d['expert_map'].to(torch.int64)
            flat=ids.flatten(); valid=(flat>=0)&(flat<em.numel()); padding=int((~valid).sum())
            valid_ids=flat[valid]; local_map=em[valid_ids] if valid_ids.numel() else torch.empty(0,dtype=torch.int64)
            local=local_map>=0; remote=int((~local).sum()); local_ids=local_map[local]
            counts=torch.bincount(local_ids,minlength=int((em>=0).sum())) if local_ids.numel() else torch.zeros(int((em>=0).sum()),dtype=torch.int64)
            active=counts[counts>0].tolist()
            pad={}
            for bm in (4,8): pad[str(bm)]=sum(math.ceil(x/bm)*bm-x for x in active)
            rows.append({'rank':rank,'call_index':idx,'chunk_index':int(d['chunk_index']),'layer_index':int(d['layer_index']),'request_id':d['request_id'],'tokens':int(d['tokens']),'routes':int(flat.numel()),'valid_global_routes':int(valid.sum()),'padding_ids':padding,'local_routes':int(local.sum()),'remote_routes':remote,'active_local_experts':len(active),'route_count_min':min(active) if active else 0,'route_count_median':pct(active,.5),'route_count_p90':pct(active,.9),'route_count_max':max(active) if active else 0,'alignment_padding_block4':pad['4'],'alignment_padding_block8':pad['8'],'fixture':str(p),'sha256':sha(p)})
        assert seen==expected,(rank,seen)
    # Same executed shapes/layers across ranks. Do not impose 1024+564 a priori.
    for idx in sorted(expected):
        pair=[r for r in rows if r['call_index']==idx]; assert len(pair)==2
        assert pair[0]['tokens']==pair[1]['tokens'] and pair[0]['layer_index']==pair[1]['layer_index']
    out={'schema':'ds41-routed-prefill-fixture-manifest-v2','status':'PASS','rows':rows}
    op=Path(a.out); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
