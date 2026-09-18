#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil, subprocess, tempfile, time
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-usable-release-001'
PROMPT=BASE/'prompts/coding-c-frame-thinking-off-v2.txt'
SPEC=json.load(open(ROOT/'runtime/ds41/daily-fixtures/c-frame-stream/task.json'))
COLLECTOR=ROOT/'scripts/run-ds4-usable-request.py'
SERVER_LOG=BASE/'coordinator.log'
OUT=BASE/'coding-c-off'
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def collect(ident,prompt_file):
 cmd=['python3',str(COLLECTOR),'--id',ident,'--out-dir',str(OUT),'--prompt-file',str(prompt_file),
      '--max-tokens','8192','--thinking','off','--reasoning-effort','none','--server-log',str(SERVER_LOG),'--timeout','1800']
 p=subprocess.run(cmd,text=True,capture_output=True);(OUT/f'{ident}.collector.stdout').write_text(p.stdout);(OUT/f'{ident}.collector.stderr').write_text(p.stderr)
 if p.returncode!=0:return None,{'status':'FAILED_TRANSPORT','rc':p.returncode,'stderr':p.stderr[-4000:]}
 return json.load(open(OUT/ident/'result.json')),None
def envelope(content):
 try:o=json.loads(content)
 except Exception as e:return None,f'invalid JSON envelope: {e}'
 if set(o)!= {'path','content'} or o.get('path')!='src/frame_parser.c' or not isinstance(o.get('content'),str):
  return None,f'envelope contract failed keys={list(o) if isinstance(o,dict) else None} path={o.get("path") if isinstance(o,dict) else None}'
 return o,None
def static_safe(src):
 bad=['system(', 'popen(', 'fork(', 'execv', 'execl', 'socket(', 'unlink(', 'remove(', '#include <unistd.h>', '#include <sys/socket.h>']
 hits=[x for x in bad if x.lower() in src.lower()]
 return hits
def test_source(src):
 base=Path(SPEC['repo'])
 if not Path('/home/funboy/StrixHaloClusterGLM/.tools/go/bin/go').exists(): pass
 with tempfile.TemporaryDirectory(prefix='ds4-usable-c-') as td:
  w=Path(td)/'repo';shutil.copytree(base,w)
  (w/'src/frame_parser.c').write_text(src)
  for dest,srcfile in SPEC['test_files'].items():
   dp=w/dest;dp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(srcfile,dp)
  env={'PATH':'/usr/bin:/bin','HOME':td,'ASAN_OPTIONS':'detect_leaks=0:abort_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
  t=time.monotonic()
  try:
   b=subprocess.run(SPEC['build_command'],cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
   out=b.stdout;trc=b.returncode
   if b.returncode==0:
    q=subprocess.run(SPEC['test_command'],cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    out+='\n'+q.stdout;trc=q.returncode
   return {'status':'PASS' if b.returncode==0 and trc==0 else 'FAIL_TEST','build_rc':b.returncode,'test_rc':trc,'seconds':time.monotonic()-t,'output_tail':out[-12000:]}
  except FileNotFoundError as e:return {'status':'BLOCKED_SYSTEM','error':repr(e)}
  except subprocess.TimeoutExpired as e:return {'status':'FAIL_TEST_TIMEOUT','error':repr(e)}
def make_repair(reason,previous,feedback=''):
 text=PROMPT.read_text()
 suffix='\n\nREPAIR CONTEXT (one authorized repair only):\n'+reason+'\n'
 if previous:suffix+='Previous final envelope/content:\n'+previous[:12000]+'\n'
 if feedback:suffix+='Executed test feedback:\n'+feedback[-12000:]+'\n'
 suffix+='Return ONLY the corrected JSON envelope with path and complete content. Do not explain.\n'
 p=OUT/'repair-prompt.txt';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text+suffix);return p
def attempt(ident,prompt):
 r,err=collect(ident,prompt)
 if err:return {'id':ident,'status':err['status'],'collector_error':err},None,None
 st=r['state']; content=st.get('content') or ''
 summary={'id':ident,'finish_reason':st.get('finish_reason'),'content_chars':len(content),'reasoning_chars':len(st.get('reasoning') or ''),'usage':st.get('usage'),'metrics':{'wall_s':r.get('wall_s'),'first_final_s':st.get('first_final_s'),'server':r.get('server'),'cache':r.get('cache')}}
 env,ee=envelope(content)
 if env is None:
  status='INCOMPLETE_NO_FINAL' if len(content)==0 and st.get('finish_reason')=='length' else 'FAIL_ENVELOPE'
  summary.update(status=status,envelope_error=ee);return summary,content,None
 hits=static_safe(env['content'])
 if hits:summary.update(status='FAIL_SAFETY_STATIC',forbidden=hits);return summary,content,None
 val=test_source(env['content']);summary['validation']=val;summary['status']=val['status'];return summary,content,val
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'terminal.json').exists():raise SystemExit('replay guard: terminal exists')
 a1,content,val=attempt('coding-c-off-attempt1',PROMPT);rows=[a1];atomic(OUT/'registry.json',{'schema':'ds4-usable-c-off-v1','attempts':rows,'updated_unix':time.time()});print(json.dumps(a1,ensure_ascii=False),flush=True)
 repair=None
 if a1['status']=='FAIL_ENVELOPE' or a1['status']=='INCOMPLETE_NO_FINAL':
  repair=make_repair('The first attempt did not produce a valid final JSON envelope.',content or '')
 elif a1['status'] in ('FAIL_TEST','FAIL_TEST_TIMEOUT') and val is not None:
  repair=make_repair('The first candidate was executed in the frozen private test harness and did not pass.',content or '',val.get('output_tail',''))
 if repair is not None:
  a2,_,_=attempt('coding-c-off-repair1',repair);rows.append(a2);atomic(OUT/'registry.json',{'schema':'ds4-usable-c-off-v1','attempts':rows,'updated_unix':time.time()});print(json.dumps(a2,ensure_ascii=False),flush=True)
 final='PASS' if rows[-1]['status']=='PASS' else rows[-1]['status']
 atomic(OUT/'terminal.json',{'schema':'ds4-usable-c-off-terminal-v1','status':final,'repairs_used':len(rows)-1,'attempts':rows,'finished_unix':time.time()})
 print('CODING_C_OFF_TERMINAL='+final,flush=True)
if __name__=='__main__':main()
