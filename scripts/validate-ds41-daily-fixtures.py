#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, subprocess, tempfile, pathlib, time
ROOT=pathlib.Path(__file__).resolve().parents[1]; BASE=ROOT/'runtime/ds41/daily-fixtures'; GO='/home/funboy/StrixHaloClusterGLM/.tools/go/bin/go'
CASES=[
 ('c-frame-stream',['gcc','-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-Iinclude','src/frame_parser.c','test.c','-o','test-bin'],['./test-bin'],{'test.c':'private/test.c'},{'src/frame_parser.c':'private/golden/src/frame_parser.c'}),
 ('c-ring-wrap',['gcc','-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-Iinclude','src/ring.c','test.c','-o','test-bin'],['./test-bin'],{'test.c':'private/test.c'},{'src/ring.c':'private/golden/src/ring.c'}),
 ('go-session-state',[GO,'test','-race','./...'],[],{'session/manager_test.go':'private/manager_test.go'},{'session/manager.go':'private/golden/session/manager.go'}),
 ('go-snapshot-feature',[GO,'test','-race','./...'],[],{'kv/snapshot_test.go':'private/snapshot_test.go'},{'kv/snapshot.go':'private/golden/kv/snapshot.go'}),
]
def run(name,golden,build,test,hidden,overrides):
 d=BASE/name
 with tempfile.TemporaryDirectory(prefix='ds41-fixture-') as td:
  w=pathlib.Path(td); shutil.copytree(d/'repo',w,dirs_exist_ok=True)
  for dest,src in hidden.items(): (w/dest).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(d/src,w/dest)
  if golden:
   for dest,src in overrides.items(): shutil.copy2(d/src,w/dest)
  env={'PATH':'/usr/bin:/bin:/home/funboy/StrixHaloClusterGLM/.tools/go/bin','HOME':td,'GOCACHE':td+'/.gocache','GOPATH':td+'/.gopath','CGO_ENABLED':'1'}
  t=time.monotonic(); p=subprocess.run(build,cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90); out=p.stdout
  rc=p.returncode
  if rc==0 and test:
   p2=subprocess.run(test,cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90); out+=p2.stdout; rc=p2.returncode
  return {'rc':rc,'pass':rc==0,'seconds':time.monotonic()-t,'output_tail':out[-2000:]}
def main():
 rows=[]
 for c in CASES:
  name,build,test,hidden,overrides=c; bad=run(name,False,build,test,hidden,overrides); good=run(name,True,build,test,hidden,overrides)
  ok=(not bad['pass']) and good['pass']; rows.append({'id':name,'buggy':bad,'golden':good,'discriminator_pass':ok}); print(name,'BUGGY',bad['rc'],'GOLDEN',good['rc'],'PASS',ok)
 out={'schema':'ds41-daily-fixture-prevalidation-v1','status':'PASS' if all(x['discriminator_pass'] for x in rows) else 'FAIL','tasks':rows}
 path=ROOT/'reports/DS41-Q2-001/daily-k2-panel/coding-fixture-prevalidation.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)+'\n')
 if out['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
