#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil, subprocess, tempfile, time, urllib.request
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=ROOT/'reports/DS41-Q2-001/recovery-upstream/r4/coding-gate5000'
URL='http://127.0.0.1:8080/v1/chat/completions'; MODEL='deepseek-v4.1-flash'
GO='/home/funboy/StrixHaloClusterGLM/.tools/go/bin/go'

def atomic(p:Path,o):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)

def prompt_for(spec:dict, repo:Path)->str:
 parts=[
  'You are fixing a small repository. Solve the task from the provided public files only.',
  'Return ONLY one valid JSON object with exactly two keys: "path" and "content".',
  '"path" must be the single allowed path. "content" must be the complete replacement file contents.',
  'Do not return markdown fences, prose, patches, commands, or additional keys.',
  'Do not modify public API unless the task explicitly requires it.',
  '', 'TASK:', spec['task'], '', 'ALLOWED PATHS: '+json.dumps(spec['allowed_paths']), '', 'PUBLIC FILES:'
 ]
 for rel in spec['files']:
  p=repo/rel; parts += [f'===== {rel} =====',p.read_text(),f'===== END {rel} =====','']
 return '\n'.join(parts)

def call(case_id:str,prompt:str,max_tokens:int=8192):
 d=OUT/case_id; d.mkdir(parents=True,exist_ok=True); sp=d/'state.json'
 if sp.exists(): raise RuntimeError(f'replay guard: {case_id} already has state')
 st={'id':case_id,'state':'IN_FLIGHT','started_unix':time.time(),'max_tokens':max_tokens,'reasoning_effort':'low'}; atomic(sp,st)
 payload={'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':1,'max_tokens':max_tokens,'stream':False,'thinking':True,'reasoning_effort':'low'}
 t=time.monotonic(); req=urllib.request.Request(URL,data=json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=1800) as resp: raw=resp.read(); code=resp.status
 except Exception as e:
  st.update(state='FAILED_TRANSPORT',finished_unix=time.time(),error=repr(e)); atomic(sp,st); return st,None
 wall=time.monotonic()-t; obj=json.loads(raw); atomic(d/'response.json',obj)
 msg=(obj.get('choices') or [{}])[0].get('message') or {}; content=msg.get('content') or ''; reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
 (d/'model-content.txt').write_text(content); (d/'model-reasoning.txt').write_text(reasoning)
 st.update(http_status=code,wall_s=wall,finish_reason=((obj.get('choices') or [{}])[0].get('finish_reason')),usage=obj.get('usage'),content_chars=len(content),reasoning_chars=len(reasoning)); atomic(sp,st)
 return st,content

def safe_source(case_id:str, text:str)->tuple[bool,str]:
 low=text.lower()
 if case_id=='coding_c_frame':
  bad=['system(', 'popen(', 'fork(', 'execv', 'execl', 'socket(', 'unlink(', 'remove(', '#include <stdlib.h>', '#include <unistd.h>', '#include <sys/socket.h>']
 else:
  bad=['"os"','"os/exec"','"net"','"net/http"','"syscall"','"unsafe"']
 hits=[x for x in bad if x in low]
 return (not hits, 'forbidden='+repr(hits))

def validate(case_id:str,spec:dict,content:str):
 d=OUT/case_id
 try: envobj=json.loads(content)
 except Exception as e: return {'status':'FAIL_ENVELOPE','error':f'invalid JSON envelope: {e}'}
 if set(envobj)!= {'path','content'} or envobj.get('path') not in spec['allowed_paths'] or not isinstance(envobj.get('content'),str):
  return {'status':'FAIL_ENVELOPE','error':'envelope keys/path/content contract failed','actual_keys':list(envobj) if isinstance(envobj,dict) else None,'path':envobj.get('path') if isinstance(envobj,dict) else None}
 allowed=envobj['path']; src=envobj['content']; ok,why=safe_source(case_id,src)
 if not ok: return {'status':'FAIL_SAFETY_STATIC','error':why,'path':allowed}
 base=ROOT/spec['repo'].removeprefix('/home/funboy/StrixHaloClusterDS41/') if str(spec['repo']).startswith('/home/funboy/StrixHaloClusterDS41/') else Path(spec['repo'])
 if not Path(base).is_absolute(): base=ROOT/base
 with tempfile.TemporaryDirectory(prefix=f'ds41-r4-{case_id}-') as td:
  w=Path(td)/'repo'; shutil.copytree(base,w)
  target=w/allowed; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(src)
  for dest,srcfile in spec['test_files'].items():
   dp=w/dest; dp.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(srcfile,dp)
  env={'PATH':'/usr/bin:/bin:/home/funboy/StrixHaloClusterGLM/.tools/go/bin','HOME':td,'GOCACHE':td+'/.gocache','GOPATH':td+'/.gopath','CGO_ENABLED':'1','ASAN_OPTIONS':'detect_leaks=0:abort_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
  build=spec['build_command']; test=spec['test_command']; t=time.monotonic()
  try:
   p=subprocess.run(build,cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
   bout=p.stdout[-12000:]; brc=p.returncode
   tout=''; trc=brc
   if brc==0 and test and test!=build:
    q=subprocess.run(test,cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120); tout=q.stdout[-12000:]; trc=q.returncode
   elif brc==0: trc=0
  except subprocess.TimeoutExpired as e:
   return {'status':'FAIL_TEST_TIMEOUT','seconds':time.monotonic()-t,'output_tail':((e.stdout or '') if isinstance(e.stdout,str) else '')[-4000:]}
  # Go spec build_command == test_command, so one successful go test is sufficient.
  passed=(brc==0 and trc==0)
  return {'status':'PASS' if passed else 'FAIL_TEST','path':allowed,'build_rc':brc,'test_rc':trc,'seconds':time.monotonic()-t,'build_output_tail':bout,'test_output_tail':tout,'replacement_bytes':len(src.encode())}

def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]
 for case_id,name in [('coding_c_frame','c-frame-stream'),('coding_go_session','go-session-state')]:
  spec=json.load(open(ROOT/f'runtime/ds41/daily-fixtures/{name}/task.json')); repo=Path(spec['repo']); prompt=prompt_for(spec,repo)
  atomic(OUT/case_id/'prompt.json',{'task_spec':str(ROOT/f'runtime/ds41/daily-fixtures/{name}/task.json'),'prompt':prompt,'allowed_paths':spec['allowed_paths'],'max_tokens':8192,'reasoning_effort':'low'})
  st,content=call(case_id,prompt,8192)
  if content is None: result={'status':st['state'],'error':st.get('error')}
  else: result=validate(case_id,spec,content)
  atomic(OUT/case_id/'validation.json',result); st=json.load(open(OUT/case_id/'state.json')); st.update(state='COMPLETE' if result.get('status')=='PASS' else 'FAILED',validation_status=result.get('status'),finished_unix=time.time()); atomic(OUT/case_id/'state.json',st)
  rows.append({'id':case_id,'state':st['state'],'validation':result}); atomic(OUT/'registry.json',{'schema':'ds41-r4-ds4-coding-v1','cases':rows,'updated_unix':time.time()}); print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
 atomic(OUT/'terminal.json',{'complete':True,'all_pass':all(r['validation'].get('status')=='PASS' for r in rows),'cases':rows,'finished_unix':time.time()}); print('CODING_COMPLETE',flush=True)
if __name__=='__main__': main()
