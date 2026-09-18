#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-usable-release-001'
OUT=BASE/'quality'
COL=ROOT/'scripts/run-ds4-usable-request.py'
SERVER=BASE/'coordinator.log'
AUD=json.load(open(BASE/'prompts/prompt-audit.json'))
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def ds4_ready():
 p=subprocess.run([str(ROOT/'scripts/ds4-usable-controller.sh'),'status'],text=True,capture_output=True)
 if p.returncode:return False
 try:return json.loads(p.stdout).get('state')=='READY'
 except:return False
def run_doc(ident,prompt,expected):
 d=OUT/ident
 if d.exists():raise RuntimeError(f'replay guard {d}')
 cmd=['python3',str(COL),'--id',ident,'--out-dir',str(OUT),'--prompt-file',str(prompt),'--max-tokens','128','--thinking','off','--reasoning-effort','none','--server-log',str(SERVER),'--timeout','1800']
 p=subprocess.run(cmd,text=True,capture_output=True);(OUT/f'{ident}.collector.stdout').write_text(p.stdout);(OUT/f'{ident}.collector.stderr').write_text(p.stderr)
 if p.returncode!=0:return {'id':ident,'status':'FAILED_TRANSPORT','rc':p.returncode,'stderr':p.stderr[-4000:]}
 r=json.load(open(d/'result.json')); content=r['state'].get('content') or ''
 try: got=json.loads(content); ok=got==expected; detail=f'expected={expected!r} actual={got!r}'
 except Exception as e: got=None;ok=False;detail=f'json parse error={e}; content={content[:1000]!r}'
 val={'id':ident,'status':'PASS' if ok else 'FAIL_SEMANTIC','semantic_pass':ok,'expected':expected,'actual':got,'detail':detail,'metrics':{'wall_s':r['wall_s'],'first_final_s':r['state'].get('first_final_s'),'cache':r.get('cache'),'server':r.get('server'),'usage':r['state'].get('usage'),'finish_reason':r['state'].get('finish_reason')}}
 atomic(d/'validation.json',val);return val
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'terminal.json').exists():raise SystemExit('replay guard terminal')
 if not ds4_ready():raise SystemExit('DS4 usable profile not READY')
 rows=[]
 for ident in ('code2k-middle-explicit-v2','docs2k-middle-explicit-v2'):
  rec=next(x for x in AUD['records'] if x['id']==ident)
  row=run_doc(ident,ROOT/rec['prompt_file'],rec['expected']);rows.append(row);atomic(OUT/'registry.json',{'schema':'ds4-usable-quality-v1','cases':rows,'updated_unix':time.time()});print(json.dumps(row,ensure_ascii=False),flush=True)
 # C runs regardless of document outcome, exactly as mandated.
 p=subprocess.run(['python3',str(ROOT/'scripts/run-ds4-usable-coding-c.py')],text=True,capture_output=True)
 (OUT/'coding-c-runner.stdout').write_text(p.stdout);(OUT/'coding-c-runner.stderr').write_text(p.stderr)
 ct=BASE/'coding-c-off/terminal.json'
 crow=json.load(open(ct)) if ct.exists() else {'status':'FAILED_RUNNER','rc':p.returncode,'stderr':p.stderr[-4000:]}
 rows.append({'id':'coding-c-frame-thinking-off-v2','status':crow.get('status'),'terminal':crow})
 docs_pass=all(x.get('status')=='PASS' for x in rows[:2])
 final={'schema':'ds4-usable-quality-terminal-v1','documents_pass':docs_pass,'coding_c_status':crow.get('status'),'cases':rows,'performance_admitted':docs_pass,'finished_unix':time.time()}
 atomic(OUT/'terminal.json',final);atomic(OUT/'registry.json',{'schema':'ds4-usable-quality-v1','cases':rows,'updated_unix':time.time()});print(json.dumps(final,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
