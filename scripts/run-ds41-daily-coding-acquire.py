#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, urllib.request
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41'); OUT=ROOT/'reports/DS41-Q2-001/daily-k2-run/coding'; TOKEN=Path('/home/funboy/.local/state/haloclu-ds41/api-token').read_text().strip()
EXPECTED_RELEASE='c075e82464c954e202c6f47658821d233edc13a5'; EXPECTED_EPOCH='1789482113148506636'
NAMES=['c-frame-stream','c-ring-wrap','go-session-state','go-snapshot-feature']
def atomic(p,o): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)
def api(method,path,obj=None,timeout=20):
    data=None if obj is None else json.dumps(obj).encode(); r=urllib.request.Request('http://127.0.0.1:18222'+path,data=data,method=method,headers={'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=timeout) as x:return json.load(x)
def check_live():
    x=api('GET','/v1/lifecycle');
    if x.get('state')!='READY' or x.get('release_id')!=EXPECTED_RELEASE or x.get('epoch')!=EXPECTED_EPOCH: raise RuntimeError('live mismatch '+json.dumps(x))
def classify(r):
    status=str(r.get('status') or ''); err=str(r.get('error') or ''); fr=r.get('final_response') or ''
    if status in ('queued','running','draining','applying'): return 'IN_FLIGHT','PENDING'
    if fr:
        if 'sandbox infrastructure' in err or status.upper()=='BLOCKED': return 'BLOCKED','MODEL_OUTPUT_ACQUIRED_TEST_BLOCKED_SYSTEM'
        if status.upper() in ('DONE','PASS','COMPLETE','SUCCEEDED'): return 'COMPLETE','PASS'
        return 'COMPLETE','MODEL_OUTPUT_ACQUIRED_TEST_NOT_PASS'
    if 'sandbox infrastructure' in err or status.upper()=='BLOCKED': return 'BLOCKED','BLOCKED_SYSTEM_NO_FINAL'
    if status.upper()=='INCOMPLETE': return 'FAILED','INCOMPLETE_NO_FINAL'
    return 'FAILED',status or 'UNKNOWN'
def reconcile(name,tid,d,st):
    r=api('GET','/v1/coding/tasks/'+tid); atomic(d/'latest.json',r); state,outcome=classify(r)
    if state!='IN_FLIGHT':
        atomic(d/'final.json',r); (d/'model-output.txt').write_text(r.get('final_response') or '')
        st.update(state=state,outcome=outcome,gateway_status=r.get('status'),error=r.get('error'),model_calls=r.get('model_calls'),wall_seconds=r.get('wall_seconds'),attempts=r.get('attempts'),finished_at=time.time(),model_output_chars=len(r.get('final_response') or ''),execution='BLOCKED_SYSTEM' if 'BLOCKED_SYSTEM' in outcome else ('EXECUTED' if r.get('status') in ('DONE','PASS','COMPLETE','SUCCEEDED') else 'NOT_EXECUTED'),correctness='NON_VERIFIED' if 'BLOCKED' in outcome or 'INCOMPLETE' in outcome else 'VERIFIED_BY_PRODUCT')
        atomic(d/'state.json',st)
    return state,r
for name in NAMES:
    d=OUT/name; d.mkdir(parents=True,exist_ok=True); sp=d/'state.json'; st=json.load(open(sp)) if sp.exists() else {'id':name,'state':'NOT_SENT'}
    if st.get('state') in ('COMPLETE','BLOCKED') or (st.get('state')=='FAILED' and st.get('task_id')):
        print(name,'reuse',st.get('state'),st.get('outcome'),flush=True); continue
    check_live()
    if st.get('state')=='IN_FLIGHT':
        tid=st['task_id']
    else:
        spec=json.load(open(ROOT/f'runtime/ds41/daily-fixtures/{name}/task.json'))
        st={'id':name,'state':'IN_FLIGHT','release':EXPECTED_RELEASE,'epoch':EXPECTED_EPOCH,'submitted_at':time.time()}; atomic(sp,st)
        r=api('POST','/v1/coding/tasks',spec); atomic(d/'submit.json',r); tid=r['id']; st['task_id']=tid; atomic(sp,st); print(name,'submitted',tid,flush=True)
    while True:
        check_live(); state,r=reconcile(name,tid,d,st); print(name,state,r.get('status'),r.get('model_calls'),flush=True)
        if state!='IN_FLIGHT': break
        time.sleep(5)
print('CODING_ACQUIRE_DONE',flush=True)
