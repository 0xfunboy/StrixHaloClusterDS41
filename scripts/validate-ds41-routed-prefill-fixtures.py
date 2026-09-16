#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(); root=Path(a.root)
 expected=[0,20,40,60]; ranks={}
 import torch
 for rank in (0,1):
  rows=[]
  for call in expected:
   pt=root/f'route-fixture-rank{rank}-call{call:02d}.pt'; js=root/f'route-fixture-rank{rank}-call{call:02d}.json'
   assert pt.exists() and js.exists(), (pt,js)
   d=torch.load(pt,map_location='cpu',weights_only=False); s=json.loads(js.read_text())
   assert d['schema']=='ds41-routed-prefill-fixture-v1' and d['rank']==rank and d['call_index']==call
   assert d['chunk_index']==call//40 and d['layer_index']==call%40
   assert d['hidden']==5120 and d['top_k']==6 and d['w13_quant']==16 and d['w2_quant']==10
   assert tuple(d['x'].shape)==(d['tokens'],5120); assert tuple(d['topk_ids'].shape)==(d['tokens'],6); assert tuple(d['topk_weights'].shape)==(d['tokens'],6)
   assert d['expert_map'] is not None and d['expert_map'].numel()==384
   assert s['tokens']==d['tokens'] and s['routes']==d['tokens']*6 and s['local_routes'] is not None
   em=d['expert_map'].to(torch.int64); ids=d['topk_ids'].to(torch.int64); local=int((em[ids]>=0).sum()); assert local==s['local_routes']
   rows.append({'call':call,'chunk':call//40,'layer':call%40,'tokens':d['tokens'],'routes':s['routes'],'local_routes':local,'active_experts':s['active_experts'],'max_routes_per_expert':s['max_routes_per_expert'],'mean_routes_per_active_expert':s['mean_routes_per_active_expert']})
  ranks[str(rank)]=rows
 assert [x['tokens'] for x in ranks['0']]==[x['tokens'] for x in ranks['1']]
 # Target activations/router decisions should be replicated after TP reductions.
 cross=[]
 for call in expected:
  a0=torch.load(root/f'route-fixture-rank0-call{call:02d}.pt',map_location='cpu',weights_only=False); a1=torch.load(root/f'route-fixture-rank1-call{call:02d}.pt',map_location='cpu',weights_only=False)
  cross.append({'call':call,'x_equal':bool(torch.equal(a0['x'],a1['x'])),'ids_equal':bool(torch.equal(a0['topk_ids'],a1['topk_ids'])),'weights_equal':bool(torch.equal(a0['topk_weights'],a1['topk_weights']))})
 out={'schema':'ds41-routed-prefill-fixture-validation-v1','status':'PASS','ranks':ranks,'cross_rank':cross,'note':'Cross-rank equality is evidence only; component tests use each rank expert_map and immutable GGUF bytes.'}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
