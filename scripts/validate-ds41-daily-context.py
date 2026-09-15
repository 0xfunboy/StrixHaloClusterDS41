#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--result',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--kind',required=True); ap.add_argument('--label',required=True); a=ap.parse_args()
r=json.load(open(a.result)); m=json.load(open(a.manifest)); meta=next(x for x in m['corpora'] if x['kind']==a.kind and x['label']==a.label)
out={'status':'FAIL','label':r.get('label'),'prompt_tokens_local_manifest':meta['actual_prompt_tokens'],'prompt_tokens_server':r.get('derived',{}).get('prompt_tokens'),'expected':meta['expected']}
try: content=json.loads(r['state']['content'])
except Exception as e: out['error']=f'final content not exact JSON: {e}'; print(json.dumps(out,indent=2)); raise SystemExit(1)
out['actual']=content
out['semantic_pass']=content==meta['expected']
d=r.get('derived',{}); out['cached_zero']=d.get('prompt_tokens_cached')==0; out['computed_present']=isinstance(d.get('prompt_tokens_computed'),(int,float)); out['prefill_present']=isinstance(d.get('pair_prefill_engine_ms_max'),(int,float)) and d['pair_prefill_engine_ms_max']>0
out['server_matches_local_token_count']=d.get('prompt_tokens')==meta['actual_prompt_tokens']
out['done']=r.get('state',{}).get('done') is True; out['finish_reason']=r.get('state',{}).get('finish_reason')
out['status']='PASS' if all([out['semantic_pass'],out['cached_zero'],out['computed_present'],out['prefill_present'],out['server_matches_local_token_count'],out['done']]) else 'FAIL'
print(json.dumps(out,indent=2)); raise SystemExit(0 if out['status']=='PASS' else 1)
