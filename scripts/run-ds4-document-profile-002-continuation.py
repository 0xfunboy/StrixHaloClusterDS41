#!/usr/bin/env python3
from __future__ import annotations
import json, os, statistics, subprocess, time
from pathlib import Path

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-document-profile-002'
Q=BASE/'quality'
OUT=BASE/'continuation'
COL=ROOT/'scripts/run-ds4-usable-request.py'
SERVER=BASE/'coordinator.log'
AUD=json.load(open(ROOT/'runtime/ds41/document-profile-002/prompt-manifest.json'))['documents']
OWNER='DS4_DOCUMENT_PROFILE_002_20260918'

def atomic(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)

def run(ident,prompt,max_tokens):
    env=os.environ.copy();env['DS4_COORDINATOR_UNIT']='ds4-document-coordinator.service';env['DS4_WORKER_UNIT']='ds4-document-worker.service'
    cmd=['python3',str(COL),'--id',ident,'--out-dir',str(OUT),'--prompt-file',str(prompt),'--max-tokens',str(max_tokens),'--thinking','on','--reasoning-effort','low','--server-log',str(SERVER),'--timeout','1800']
    p=subprocess.run(cmd,text=True,capture_output=True,env=env)
    (OUT/f'{ident}.stdout').write_text(p.stdout);(OUT/f'{ident}.stderr').write_text(p.stderr)
    if p.returncode:return {'id':ident,'status':'FAILED_TRANSPORT','rc':p.returncode}
    return {'id':ident,'status':'COMPLETE','result':json.load(open(OUT/ident/'result.json'))}

def validate(row,exp):
    if row['status']!='COMPLETE':return row
    r=row['result'];s=r['state'];content=s.get('content') or ''
    try:got=json.loads(content);ok=got==exp
    except:got=None;ok=False
    cache=r.get('cache') or {};ind=isinstance(cache.get('cached_tokens'),int) and cache['cached_tokens']<=32
    return {'id':row['id'],'status':'PASS' if ok else 'FAIL_SEMANTIC','semantic_pass':ok,'independent':ind,'expected':exp,'actual':got,'wall_s':r.get('wall_s'),'first_final_s':s.get('first_final_s'),'first_reasoning_s':s.get('first_reasoning_s'),'finish_reason':s.get('finish_reason'),'usage':s.get('usage'),'cache':cache,'server':r.get('server'),'resources_before':r.get('resources_before'),'resources_after':r.get('resources_after')}

def rec(ident):return next(x for x in AUD['records'] if x['id']==ident)

def metric_from_quality(caseid):
    q=json.load(open(Q/'terminal.json'))
    x=next(c for c in q['cases'] if c['id']==caseid)
    return x

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'terminal.json').exists():raise SystemExit('replay guard')
    qt=json.load(open(Q/'terminal.json'))
    if not qt.get('six_pass'):raise SystemExit('six-pass quality gate absent')
    code=rec('code2k-middle-explicit-v2')
    # Reuse original and confirmation quality requests as code samples 1/2.
    qcode=next(c for c in qt['cases'] if c['id']=='doc002-code2k-middle-explicit-v2')
    qconfirm=next(c for c in qt['cases'] if c['id']=='doc002-code2k-middle-explicit-v2-confirm')
    samples=[qcode,qconfirm]
    # One additional comparable code sample, not a fresh three-run campaign.
    third=validate(run('doc002-perf-code2k-third',ROOT/code['prompt_file'],2048),code['expected']);samples.append(third)
    def prefill_tps(x):
        try:return x['metrics']['server']['prefill']['avg_tps']
        except:
            try:return x['server']['prefill']['avg_tps']
            except:return None
    valid=[x for x in samples if x.get('status')=='PASS' and x.get('independent',True)]
    vals=[prefill_tps(x) for x in valid];vals=[v for v in vals if isinstance(v,(int,float))]
    perf={'samples':samples,'valid_independent_count':len(valid),'prefill_tps':vals,'median_prefill_tps':statistics.median(vals) if vals else None,'sample_sd':statistics.stdev(vals) if len(vals)>1 else 0 if vals else None,'target200_met':bool(vals) and statistics.median(vals)>=200}
    atomic(OUT/'performance.json',perf)
    contexts=[]
    for ident in ('code4k-middle-explicit-v2','code8k-middle-explicit-v2','code16k-middle-explicit-v2'):
        r=rec(ident)
        row=validate(run('doc002-'+ident,ROOT/r['prompt_file'],512),r['expected']);contexts.append(row)
        if row.get('status')!='PASS':break
    term={'schema':'ds4-document-profile-002-continuation-v1','owner':OWNER,'performance':perf,'contexts':contexts,'max_context_pass':next((x['id'] for x in reversed(contexts) if x.get('status')=='PASS'),None),'product_admitted':bool(contexts) and contexts[0].get('status')=='PASS','finished_unix':time.time()}
    atomic(OUT/'terminal.json',term);print(json.dumps(term,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
