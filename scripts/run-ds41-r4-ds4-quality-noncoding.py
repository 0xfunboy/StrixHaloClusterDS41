#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, urllib.request, urllib.error
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=ROOT/'reports/DS41-Q2-001/recovery-upstream/r4/quality'
URL='http://127.0.0.1:8080/v1/chat/completions'
MODEL='deepseek-v4.1-flash'

def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)
def request(case):
 d=OUT/case['id']; d.mkdir(parents=True,exist_ok=True); sp=d/'state.json'; rp=d/'response.json'
 if sp.exists():
  old=json.load(open(sp));
  if old.get('state') in ('COMPLETE','FAILED','IN_FLIGHT'): raise RuntimeError(f"refuse replay {case['id']} state={old.get('state')}")
 st={'id':case['id'],'state':'IN_FLIGHT','started_unix':time.time(),'expected':case.get('expected'),'messages_sha256':None}
 atomic(sp,st)
 payload={'model':MODEL,'messages':case['messages'],'temperature':0,'seed':1,'max_tokens':case.get('max_tokens',128),'stream':False,'thinking':False,'reasoning_effort':'none'}
 body=json.dumps(payload,separators=(',',':')).encode(); t=time.monotonic()
 req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=900) as resp:
   raw=resp.read(); code=resp.status
 except Exception as e:
  st.update(state='FAILED',finished_unix=time.time(),error=repr(e)); atomic(sp,st); return st
 wall=time.monotonic()-t
 try: obj=json.loads(raw)
 except Exception as e:
  obj={'_raw':raw.decode(errors='replace'),'_json_error':repr(e)}
 atomic(rp,obj)
 content=''
 reasoning=''
 try:
  msg=obj['choices'][0]['message']; content=msg.get('content') or ''; reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
 except Exception: pass
 ok=False; detail=''
 try:
  typ=case['validator']
  if typ=='json':
   got=json.loads(content); ok=(got==case['expected']); detail=f'expected={case["expected"]!r} actual={got!r}'
  elif typ=='exact':
   got=content.strip(); ok=(got==case['expected']); detail=f'expected={case["expected"]!r} actual={got!r}'
  else: raise ValueError(typ)
 except Exception as e: detail=f'validation error: {e}; content={content[:1000]!r}'
 st.update(state='COMPLETE' if ok else 'FAILED',finished_unix=time.time(),http_status=code,wall_s=wall,semantic_pass=ok,detail=detail,content=content,reasoning=reasoning,usage=obj.get('usage'),finish_reason=((obj.get('choices') or [{}])[0].get('finish_reason')))
 atomic(sp,st); return st

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 man=json.load(open(ROOT/'reports/DS41-Q2-001/daily-k2-panel/corpora/manifest.json'))
 code_exp=next(x['expected'] for x in man['corpora'] if x['kind']=='code' and x['label']=='2k')
 docs_exp=next(x['expected'] for x in man['corpora'] if x['kind']=='docs' and x['label']=='2k')
 discr=json.load(open(ROOT/'reports/DS41-Q2-001/prefill-priority-001/code2k-extract-preregister.json'))['expected']
 hold=json.load(open(ROOT/'runtime/ds41/recovery-holdouts-v1.json'))['cases']
 cases=[
  {'id':'code2k1588','messages':[{'role':'user','content':(ROOT/'reports/DS41-Q2-001/daily-k2-panel/corpora/code-2k.txt').read_text()}],'expected':code_exp,'validator':'json','max_tokens':128},
  {'id':'tail1546','messages':[{'role':'user','content':(ROOT/'reports/DS41-Q2-001/prefill-priority-001/code2k-tail-only.txt').read_text()}],'expected':{'end':5},'validator':'json','max_tokens':128},
  {'id':'discriminator1571','messages':[{'role':'user','content':(ROOT/'reports/DS41-Q2-001/prefill-priority-001/code2k-extract.txt').read_text()}],'expected':discr,'validator':'json','max_tokens':256},
  {'id':'docs2k','messages':[{'role':'user','content':(ROOT/'reports/DS41-Q2-001/daily-k2-panel/corpora/docs-2k.txt').read_text()}],'expected':docs_exp,'validator':'json','max_tokens':128},
 ]
 for h in hold:
  cases.append({'id':h['id'],'messages':h['messages'],'expected':h['expected_exact'],'validator':'exact','max_tokens':h['sampling']['max_tokens']})
 cases += [
  {'id':'json_none','messages':[{'role':'user','content':'Return exactly this JSON object and no prose: {"ok":true,"n":7}'}],'expected':{'ok':True,'n':7},'validator':'json','max_tokens':128},
  {'id':'fresh_marker','messages':[{'role':'user','content':'Return exactly STABLE-01.'}],'expected':'STABLE-01','validator':'exact','max_tokens':128},
  {'id':'multiturn_orbit','messages':[{'role':'user','content':'Remember codeword ORBIT-17.'},{'role':'assistant','content':'Acknowledged.'},{'role':'user','content':'Return only the codeword.'}],'expected':'ORBIT-17','validator':'exact','max_tokens':128},
 ]
 rows=[]
 for c in cases:
  try: st=request(c)
  except Exception as e: st={'id':c['id'],'state':'BLOCKED_REPLAY_GUARD','error':repr(e)}
  rows.append(st); atomic(OUT/'registry.json',{'schema':'ds41-r4-ds4-quality-v1','updated_unix':time.time(),'cases':rows})
  print(json.dumps({'id':st.get('id'),'state':st.get('state'),'semantic_pass':st.get('semantic_pass'),'wall_s':st.get('wall_s'),'detail':st.get('detail')},ensure_ascii=False),flush=True)
 atomic(OUT/'noncoding-terminal.json',{'schema':'ds41-r4-ds4-quality-noncoding-v1','complete':True,'all_pass':all(x.get('semantic_pass') is True for x in rows),'cases':rows,'finished_unix':time.time()})
 print('NONCODING_COMPLETE',flush=True)
if __name__=='__main__': main()
