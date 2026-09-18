#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, urllib.request
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41'); OUT=ROOT/'reports/DS41-Q2-001/recovery-upstream/r4/quality-gate5000'; URL='http://127.0.0.1:8080/v1/chat/completions'; MODEL='deepseek-v4.1-flash'
def atomic(p,o): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)
def run(c):
 d=OUT/c['id']; d.mkdir(parents=True,exist_ok=True); sp=d/'state.json'; rp=d/'response.json'
 if sp.exists(): raise RuntimeError(f'replay guard {c["id"]}')
 st={'id':c['id'],'state':'IN_FLIGHT','started_unix':time.time(),'expected':c['expected']}; atomic(sp,st)
 payload={'model':MODEL,'messages':c['messages'],'temperature':0,'seed':1,'max_tokens':c.get('max_tokens',128),'stream':False,'thinking':False,'reasoning_effort':'none'}
 t=time.monotonic(); req=urllib.request.Request(URL,data=json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=900) as resp: raw=resp.read(); code=resp.status
 except Exception as e: st.update(state='FAILED_TRANSPORT',finished_unix=time.time(),error=repr(e)); atomic(sp,st); return st
 wall=time.monotonic()-t; obj=json.loads(raw); atomic(rp,obj)
 msg=(obj.get('choices') or [{}])[0].get('message') or {}; content=msg.get('content') or ''; reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
 ok=False
 try:
  if c['validator']=='json': got=json.loads(content); ok=got==c['expected']
  else: got=content.strip(); ok=got==c['expected']
  detail=f'expected={c["expected"]!r} actual={got!r}'
 except Exception as e: detail=f'validation error {e}; content={content[:1000]!r}'
 st.update(state='COMPLETE' if ok else 'FAILED_SEMANTIC',finished_unix=time.time(),http_status=code,wall_s=wall,semantic_pass=ok,detail=detail,content=content,reasoning=reasoning,usage=obj.get('usage'),finish_reason=((obj.get('choices') or [{}])[0].get('finish_reason'))); atomic(sp,st); return st
def main():
 hold=json.load(open(ROOT/'runtime/ds41/recovery-holdouts-v1.json'))['cases']; cases=[]
 for h in hold: cases.append({'id':h['id'],'messages':h['messages'],'expected':h['expected_exact'],'validator':'exact','max_tokens':h['sampling']['max_tokens']})
 cases += [
 {'id':'json_none','messages':[{'role':'user','content':'Return exactly this JSON object and no prose: {"ok":true,"n":7}'}],'expected':{'ok':True,'n':7},'validator':'json','max_tokens':128},
 {'id':'fresh_marker','messages':[{'role':'user','content':'Return exactly STABLE-01.'}],'expected':'STABLE-01','validator':'exact','max_tokens':128},
 {'id':'multiturn_orbit','messages':[{'role':'user','content':'Remember codeword ORBIT-17.'},{'role':'assistant','content':'Acknowledged.'},{'role':'user','content':'Return only the codeword.'}],'expected':'ORBIT-17','validator':'exact','max_tokens':128}]
 rows=[]
 for c in cases:
  st=run(c); rows.append(st); atomic(OUT/'registry.json',{'schema':'ds41-r4-ds4-quality-gate5000-v1','updated_unix':time.time(),'cases':rows}); print(json.dumps({'id':st['id'],'state':st['state'],'semantic_pass':st.get('semantic_pass'),'wall_s':st.get('wall_s'),'detail':st.get('detail'),'error':st.get('error')},ensure_ascii=False),flush=True)
 atomic(OUT/'terminal.json',{'complete':True,'all_pass':all(x.get('semantic_pass') is True for x in rows),'cases':rows,'finished_unix':time.time()}); print('RESUME_COMPLETE',flush=True)
if __name__=='__main__': main()
