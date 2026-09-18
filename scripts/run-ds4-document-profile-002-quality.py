#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-document-profile-002'
OUT=BASE/'quality'
COL=ROOT/'scripts/run-ds4-usable-request.py'
SERVER=BASE/'coordinator.log'
OWNER='DS4_DOCUMENT_PROFILE_002_20260918'
MANIFEST=json.load(open(ROOT/'runtime/ds41/document-profile-002/prompt-manifest.json'))
DOC_AUD=MANIFEST['documents']
HOLDS=MANIFEST['holdouts']

def atomic(p:Path,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)

def ready():
    p=subprocess.run([str(ROOT/'scripts/ds4-document-controller.sh'),'status'],text=True,capture_output=True)
    if p.returncode: return False
    try:
        o=json.loads(p.stdout); return o.get('state')=='READY' and o.get('owner')==OWNER
    except: return False

def run_case(ident:str,prompt:Path,expected:dict,independent_required=False):
    d=OUT/ident
    if d.exists(): raise RuntimeError(f'replay guard {ident}')
    env=os.environ.copy()
    env['DS4_COORDINATOR_UNIT']='ds4-document-coordinator.service'
    env['DS4_WORKER_UNIT']='ds4-document-worker.service'
    cmd=['python3',str(COL),'--id',ident,'--out-dir',str(OUT),'--prompt-file',str(prompt),
         '--max-tokens','2048','--thinking','on','--reasoning-effort','low',
         '--server-log',str(SERVER),'--timeout','1800']
    p=subprocess.run(cmd,text=True,capture_output=True,env=env)
    (OUT/f'{ident}.collector.stdout').write_text(p.stdout)
    (OUT/f'{ident}.collector.stderr').write_text(p.stderr)
    if p.returncode!=0:
        row={'id':ident,'status':'FAILED_TRANSPORT','rc':p.returncode,'stderr':p.stderr[-4000:]}
        atomic(d/'validation.json',row); return row
    r=json.load(open(d/'result.json'))
    st=r['state']; content=st.get('content') or ''
    finish=st.get('finish_reason')
    try:
        got=json.loads(content)
        ok=got==expected
        detail=f'expected={expected!r} actual={got!r}'
    except Exception as e:
        got=None; ok=False; detail=f'parse error={e}; content={content[:2000]!r}'
    incomplete=(finish=='length' and not content.strip())
    cached=(r.get('cache') or {}).get('cached_tokens')
    independent=(isinstance(cached,int) and cached<=32)
    status='PASS' if ok else ('INCOMPLETE' if incomplete else 'FAIL_SEMANTIC')
    if status=='PASS' and independent_required and not independent:
        status='FAIL_INDEPENDENCE'
    row={
      'id':ident,'status':status,'semantic_pass':ok,'independent':independent,
      'independent_required':independent_required,'expected':expected,'actual':got,'detail':detail,
      'profile':{'thinking':True,'reasoning_effort':'low','resolved_mode':'DS4_THINK_LOW','numeric_level':None,'max_tokens_total':2048,'temperature':0,'seed':1},
      'metrics':{
        'wall_s':r.get('wall_s'),'headers_s':r.get('headers_s'),
        'first_any_s':st.get('first_any_s'),'first_reasoning_s':st.get('first_reasoning_s'),'first_final_s':st.get('first_final_s'),
        'reasoning_chars':len(st.get('reasoning') or ''),'final_chars':len(content),
        'finish_reason':finish,'usage':st.get('usage'),'cache':r.get('cache'),'server':r.get('server'),
        'resources_before':r.get('resources_before'),'resources_after':r.get('resources_after')
      }
    }
    atomic(d/'validation.json',row); return row

def rec(ident):
    return next(x for x in DOC_AUD['records'] if x['id']==ident)
def hold(ident):
    return next(x for x in HOLDS if x['id']==ident)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    terminal=OUT/'terminal.json'
    if terminal.exists(): raise SystemExit('replay guard: terminal exists')
    if not ready(): raise SystemExit('DOCUMENT_PROFILE_002 not READY')
    rows=[]
    def add(row):
        rows.append(row)
        atomic(OUT/'registry.json',{'schema':'ds4-document-profile-002-quality-v1','owner':OWNER,'requests_used':len(rows),'max_requests':6,'cases':rows,'updated_unix':time.time()})
        print(json.dumps(row,ensure_ascii=False),flush=True)

    for ident in ('code2k-middle-explicit-v2','docs2k-middle-explicit-v2'):
        x=rec(ident); add(run_case('doc002-'+ident,ROOT/x['prompt_file'],x['expected']))
    originals_pass=all(x['status']=='PASS' for x in rows[:2])
    if originals_pass:
        for ident in ('document-holdout-a','document-holdout-b'):
            x=hold(ident); add(run_case('doc002-'+ident,ROOT/x['prompt_file'],x['expected']))
    holdouts_pass=originals_pass and len(rows)>=4 and all(x['status']=='PASS' for x in rows[2:4])
    if holdouts_pass:
        for ident in ('code2k-middle-explicit-v2','docs2k-middle-explicit-v2'):
            x=rec(ident); add(run_case('doc002-'+ident+'-confirm',ROOT/x['prompt_file'],x['expected'],independent_required=True))
    six_pass=len(rows)==6 and all(x['status']=='PASS' for x in rows)
    status='DOCUMENT_PROFILE_QUALITY_PASS' if six_pass else 'DOCUMENT_PROFILE_NOT_QUALIFIED'
    term={'schema':'ds4-document-profile-002-quality-terminal-v1','owner':OWNER,'status':status,'requests_used':len(rows),'budget_max':6,'originals_pass':originals_pass,'holdouts_pass':holdouts_pass,'six_pass':six_pass,'cases':rows,'finished_unix':time.time()}
    atomic(terminal,term)
    print(json.dumps(term,ensure_ascii=False),flush=True)
    if not six_pass:
        subprocess.run([str(ROOT/'scripts/finalize-ds4-document-profile-002.sh'),'NOT_QUALIFIED'],check=True)
    else:
        atomic(BASE/'continuation-admitted.json',{'owner':OWNER,'state':'ADMITTED','reason':'six document requests PASS','updated_unix':time.time()})
if __name__=='__main__': main()
