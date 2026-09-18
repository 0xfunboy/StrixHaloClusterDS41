#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, statistics, subprocess, time
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-usable-release-001'; OUT=BASE/'performance-context'
COL=ROOT/'scripts/run-ds4-usable-request.py'; SERVER=BASE/'coordinator.log'
AUD=json.load(open(BASE/'prompts/prompt-audit.json'))
EXPECTED={x['id']:x['expected'] for x in AUD['records']}
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def run(ident,prompt,max_tokens=128):
 d=OUT/ident
 if d.exists():raise RuntimeError(f'replay guard {ident}')
 cmd=['python3',str(COL),'--id',ident,'--out-dir',str(OUT),'--prompt-file',str(prompt),'--max-tokens',str(max_tokens),'--thinking','off','--reasoning-effort','none','--server-log',str(SERVER),'--timeout','1800']
 p=subprocess.run(cmd,text=True,capture_output=True);(OUT/f'{ident}.collector.stdout').write_text(p.stdout);(OUT/f'{ident}.collector.stderr').write_text(p.stderr)
 if p.returncode!=0:return {'id':ident,'status':'FAILED_TRANSPORT','rc':p.returncode}
 r=json.load(open(d/'result.json'));return {'id':ident,'status':'COMPLETE','result':r}
def validate(row,expected):
 if row['status']!='COMPLETE':return row
 r=row['result'];content=r['state'].get('content') or ''
 try:got=json.loads(content);ok=got==expected
 except Exception as e:got=None;ok=False
 cache=r.get('cache') or {};pt=cache.get('prompt_tokens');cached=cache.get('cached_tokens')
 independent=isinstance(cached,int) and cached<=32 and isinstance(pt,int) and pt>0
 return {'id':row['id'],'status':'PASS' if ok else 'FAIL_SEMANTIC','semantic_pass':ok,'actual':got,'expected':expected,'independent_prefill':independent,'cache':cache,'wall_s':r.get('wall_s'),'first_final_s':r['state'].get('first_final_s'),'first_reasoning_s':r['state'].get('first_reasoning_s'),'finish_reason':r['state'].get('finish_reason'),'usage':r['state'].get('usage'),'server':r.get('server'),'resources_before':r.get('resources_before'),'resources_after':r.get('resources_after')}
def reset(tag):
 p=OUT/f'{tag}.txt';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f'Return exactly RESET-{tag}.')
 return run(f'reset-{tag}',p,16)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'terminal.json').exists():raise SystemExit('replay guard terminal')
 q=json.load(open(BASE/'quality/terminal.json'))
 if not q.get('documents_pass'):raise SystemExit('documents quality gate not PASS')
 code=next(x for x in AUD['records'] if x['id']=='code2k-middle-explicit-v2'); docs=next(x for x in AUD['records'] if x['id']=='docs2k-middle-explicit-v2')
 # excluded warmup/reset
 warm=reset('WARMUP');atomic(OUT/'warmup.json',warm)
 samples=[]
 for n in (1,2,3):
  reset(f'P{n}')
  row=validate(run(f'P{n}-code2k-v2',ROOT/code['prompt_file']),code['expected']);samples.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  atomic(OUT/'registry.json',{'schema':'ds4-usable-perf-context-v1','performance_samples':samples,'updated_unix':time.time()})
 perf_valid=[x for x in samples if x.get('status')=='PASS' and x.get('independent_prefill')]
 vals=[x['server'].get('prefill',{}).get('avg_tps') for x in perf_valid]; vals=[x for x in vals if isinstance(x,(int,float))]
 secs=[x['server'].get('prefill',{}).get('engine_s') for x in perf_valid];secs=[x for x in secs if isinstance(x,(int,float))]
 dec=[x['server'].get('decode',{}).get('avg_tps') for x in perf_valid];dec=[x for x in dec if isinstance(x,(int,float))]
 summary={'samples_total':len(samples),'independent_passing':len(perf_valid),'prefill_tps_samples':vals,'prefill_s_samples':secs,'decode_tps_samples':dec,'prefill_tps_median':statistics.median(vals) if vals else None,'prefill_tps_sample_sd':statistics.stdev(vals) if len(vals)>1 else 0 if vals else None,'prefill_s_median':statistics.median(secs) if secs else None,'decode_tps_median':statistics.median(dec) if dec else None,'target200_met':bool(vals) and statistics.median(vals)>=200}
 atomic(OUT/'performance-summary.json',summary)
 # docs core confirmation after reset
 reset('DOCS-CONFIRM')
 docs_conf=validate(run('docs2k-v2-confirm',ROOT/docs['prompt_file']),docs['expected']);atomic(OUT/'docs-confirm.json',docs_conf);print(json.dumps(docs_conf,ensure_ascii=False),flush=True)
 contexts=[]
 if docs_conf.get('status')=='PASS':
  for ident in ('code4k-middle-explicit-v2','code8k-middle-explicit-v2','code16k-middle-explicit-v2'):
   rec=next(x for x in AUD['records'] if x['id']==ident);reset('CTX-'+ident.split('-')[0].upper())
   row=validate(run(ident,ROOT/rec['prompt_file']),rec['expected']);contexts.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
   if row.get('status')!='PASS':break
 final={'schema':'ds4-usable-performance-context-terminal-v1','performance':summary,'samples':samples,'docs_confirmation':docs_conf,'contexts':contexts,'max_context_pass':next((x['id'] for x in reversed(contexts) if x.get('status')=='PASS'),None),'finished_unix':time.time()}
 atomic(OUT/'terminal.json',final);print(json.dumps(final,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
