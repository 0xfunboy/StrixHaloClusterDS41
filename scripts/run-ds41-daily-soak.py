#!/usr/bin/env python3
from __future__ import annotations
import argparse,http.client,json,os,socket,threading,time
from pathlib import Path
from urllib.parse import urlparse
MODEL='DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2'
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+'.tmp');t.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def utc():
 import datetime;return datetime.datetime.now(datetime.timezone.utc).isoformat()
def reqjson(base,tok,method,path,body=None,timeout=10):
 u=urlparse(base);c=http.client.HTTPConnection(u.hostname,u.port,timeout=timeout);b=None if body is None else json.dumps(body,separators=(',',':')).encode();h={'Authorization':'Bearer '+tok}
 if b is not None:h['Content-Type']='application/json'
 c.request(method,path,body=b,headers=h);r=c.getresponse();raw=r.read();s=r.status;c.close()
 try:d=json.loads(raw)
 except:d={'raw':raw.decode(errors='replace')[:4096]}
 return s,d
def lifecycle(base,tok):
 s,d=reqjson(base,tok,'GET','/v1/status',timeout=5)
 if s!=200:raise RuntimeError(f'status HTTP{s}')
 lc=d.get('lifecycle')
 if not isinstance(lc,dict):raise RuntimeError('status missing lifecycle')
 return lc
def health():
 u=urlparse('http://127.0.0.1:18221');c=http.client.HTTPConnection(u.hostname,u.port,timeout=5);c.request('GET','/health');r=c.getresponse();d=json.loads(r.read());c.close();return d
def snapshot(base,tok):
 s,d=reqjson(base,tok,'GET','/v1/status',timeout=10);return {'time':utc(),'http':s,'lifecycle':d.get('lifecycle'),'health':d.get('health'),'nodes':d.get('nodes'),'active_request':d.get('active_request')}
def parse_frame(lines,st,now,start):
 if not lines:return
 event='message';data=[]
 for x in lines:
  if x.startswith('event:'):event=x[6:].strip()
  elif x.startswith('data:'):data.append(x[5:].lstrip())
 if not data:return
 txt='\n'.join(data)
 if event=='haloclu.timing':
  try:st['gateway_timing'].append(json.loads(txt))
  except:pass
  return
 if txt=='[DONE]':st['done']=True;return
 try:o=json.loads(txt)
 except:return
 if o.get('usage') is not None:st['usage']=o['usage']
 if o.get('metrics') is not None:st['metrics']=o['metrics']
 for ch in o.get('choices') or []:
  d=ch.get('delta') or {};rr=d.get('reasoning') or d.get('reasoning_content') or '';cc=d.get('content') or ''
  if (rr or cc) and st['first_any_s'] is None:st['first_any_s']=now-start
  if cc and st['first_final_s'] is None:st['first_final_s']=now-start
  st['reasoning']+=rr;st['content']+=cc
  if ch.get('finish_reason') is not None:st['finish_reason']=ch.get('finish_reason');st['stop_reason']=ch.get('stop_reason')
def send(base,tok,messages,reasoning='none',max_tokens=128,timeout=1800,cancel_after=None,raw=None):
 u=urlparse(base);c=http.client.HTTPConnection(u.hostname,u.port,timeout=timeout)
 p={'model':MODEL,'messages':messages,'temperature':0,'seed':1,'stream':True,'stream_options':{'include_usage':True,'continuous_usage_stats':True},'max_tokens':max_tokens,'context_tokens':4096,'reasoning_effort':reasoning,'chat_template_kwargs':{'reasoning_effort':reasoning}}
 start=time.monotonic();c.request('POST','/v1/chat/completions',body=json.dumps(p,separators=(',',':')).encode(),headers={'Authorization':'Bearer '+tok,'Content-Type':'application/json','Accept':'text/event-stream','X-HaloClu-Timings':'1'});r=c.getresponse();hs=time.monotonic()-start
 st={'done':False,'reasoning':'','content':'','usage':None,'metrics':None,'finish_reason':None,'stop_reason':None,'first_any_s':None,'first_final_s':None,'gateway_timing':[]};rows=[];fr=[];cancelled=False
 if r.status!=200:
  body=r.read(8192).decode(errors='replace');c.close();return {'http_status':r.status,'error':body,'wall_s':time.monotonic()-start,'state':st,'cancelled':False}
 if cancel_after is not None and c.sock is not None:c.sock.settimeout(.5)
 while True:
  try:b=r.readline()
  except socket.timeout:
   if cancel_after is not None and time.monotonic()-start>=cancel_after:cancelled=True;break
   continue
  if not b:break
  x=b.decode('utf-8',errors='strict').rstrip('\r\n');rows.append(x)
  if x=='':
   parse_frame(fr,st,time.monotonic(),start);fr=[]
   if st['done']:break
  else:fr.append(x)
  if cancel_after is not None and time.monotonic()-start>=cancel_after:cancelled=True;break
 try:c.close()
 except:pass
 if raw:Path(raw).write_text('\n'.join(rows)+'\n')
 return {'http_status':r.status,'headers_s':hs,'wall_s':time.monotonic()-start,'state':st,'cancelled':cancelled}
def wait_drain(limit=900):
 t=time.monotonic();seen=False
 while time.monotonic()-t<limit:
  h=health();seen=seen or bool(h.get('busy'))
  if not h.get('busy'):return {'drain_s':time.monotonic()-t,'busy_seen':seen,'health':h}
  time.sleep(1)
 return {'drain_s':time.monotonic()-t,'busy_seen':seen,'timeout':True,'health':health()}
def validate(case,res):
 st=res.get('state',{});content=(st.get('content') or '').strip();service='PASS' if res.get('http_status')==200 and (st.get('done') or res.get('cancelled')) else 'FAIL';sem='N/A';detail=''
 if 'expected_exact' in case:sem='PASS' if content==case['expected_exact'] else 'FAIL';detail=f'expected={case["expected_exact"]!r} actual={content!r}'
 elif 'expected_json' in case:
  try:v=json.loads(content);sem='PASS' if v==case['expected_json'] else 'FAIL';detail=f'actual={v!r}'
  except Exception as e:sem='FAIL';detail=str(e)
 elif 'min_final_tokens' in case:
  u=st.get('usage') or {};rt=(u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0;ft=(u.get('completion_tokens') or 0)-rt;sem='PASS' if ft>=case['min_final_tokens'] else 'FAIL';detail=f'final_tokens={ft} min={case["min_final_tokens"]}'
 return service,sem,detail
def filler(label,n=420):return [{'role':'user','content':' '.join(f'w{i%97}' for i in range(n))+f'\nSENTINEL={label}\nReturn only the sentinel value.'}]
def cancel_case(case,base,tok,out):
 r=send(base,tok,case['messages'],case.get('reasoning','none'),case.get('max_tokens',2048),cancel_after=case.get('cancel_after',5),raw=out/'response.partial.sse');r['drain']=wait_drain();return r
def busy_case(case,base,tok,out):
 holder={}
 def primary():holder['r']=send(base,tok,case['messages'],'none',1024,raw=out/'primary.sse')
 th=threading.Thread(target=primary,daemon=True);th.start();t=time.monotonic()
 while time.monotonic()-t<30:
  if health().get('busy'):break
  time.sleep(.2)
 try:s=send(base,tok,[{'role':'user','content':'Return exactly SECONDARY.'}],'none',32,timeout=3,raw=out/'secondary.sse');sec={'kind':'returned','http_status':s.get('http_status'),'done':s.get('state',{}).get('done'),'content':s.get('state',{}).get('content')}
 except Exception as e:sec={'kind':'timeout_or_refused','error':repr(e)}
 th.join(1200);r=holder.get('r',{'http_status':0,'error':'primary did not finish'});r['secondary']=sec;return r
def cases():
 return [
 {'id':'01-arith-none','messages':[{'role':'user','content':'Return only the integer result of 17*19.'}],'expected_exact':'323'},
 {'id':'02-json-none','messages':[{'role':'user','content':'Return exactly this JSON object and no prose: {"ok":true,"n":7}'}],'expected_json':{'ok':True,'n':7}},
 {'id':'03-fresh-marker','messages':[{'role':'user','content':'Return exactly STABLE-01.'}],'expected_exact':'STABLE-01'},
 {'id':'04-multiturn-orbit','messages':[{'role':'user','content':'Remember codeword ORBIT-17.'},{'role':'assistant','content':'Acknowledged.'},{'role':'user','content':'Return only the codeword.'}],'expected_exact':'ORBIT-17'},
 {'id':'05-multiturn-sum','messages':[{'role':'user','content':'What is 5+7?'},{'role':'assistant','content':'12'},{'role':'user','content':'Return only the previous numeric result.'}],'expected_exact':'12'},
 {'id':'06-low-doors','reasoning':'low','max_tokens':256,'messages':[{'role':'user','content':'Classic 100 doors puzzle. Return only the number of open doors.'}],'expected_exact':'10'},
 {'id':'07-high-arith','reasoning':'high','max_tokens':512,'messages':[{'role':'user','content':'Return only the integer result of 37*41.'}],'expected_exact':'1517'},
 {'id':'08-two-lines','messages':[{'role':'user','content':'Return exactly two lines, first alpha and second beta, no other text.'}],'expected_exact':'alpha\nbeta'},
 {'id':'09-sentinel-short','messages':filler('EMBER-509'),'expected_exact':'EMBER-509'},
 {'id':'10-json-array','messages':[{'role':'user','content':'Return exactly JSON: {"items":[1,2,3],"stable":true}'}],'expected_json':{'items':[1,2,3],'stable':True}},
 {'id':'11-long-output','max_tokens':768,'messages':[{'role':'user','content':'Write a numbered list from 1 through 180. Each line must be exactly N: stable where N is the line number. No preface or epilogue.'}],'min_final_tokens':512},
 {'id':'12-cancel-decode','kind':'cancel','cancel_after':8,'max_tokens':2048,'messages':[{'role':'user','content':'Write a numbered list from 1 through 1000, one item per line, with a short unique sentence for every number.'}]},
 {'id':'13-post-cancel','messages':[{'role':'user','content':'Return only 81.'}],'expected_exact':'81'},
 {'id':'14-cancel-prefill-short','kind':'cancel','cancel_after':2,'max_tokens':64,'messages':filler('PREFILL-CANCEL',800)},
 {'id':'15-post-prefill-cancel','messages':[{'role':'user','content':'Return exactly RESUMED.'}],'expected_exact':'RESUMED'},
 {'id':'16-busy-admission','kind':'busy','messages':[{'role':'user','content':'Write 250 numbered lines, each containing the word busycheck and its line number.'}]},
 {'id':'17-post-busy','messages':[{'role':'user','content':'Return exactly SAFE.'}],'expected_exact':'SAFE'},
 {'id':'18-arith-two','messages':[{'role':'user','content':'Return only the integer result of 13*17.'}],'expected_exact':'221'},
 {'id':'19-low-short','reasoning':'low','max_tokens':192,'messages':[{'role':'user','content':'Return only the integer result of 6*7.'}],'expected_exact':'42'},
 {'id':'20-high-short','reasoning':'high','max_tokens':384,'messages':[{'role':'user','content':'Return only the integer result of 7*8.'}],'expected_exact':'56'},
 {'id':'21-multiturn-nova','messages':[{'role':'user','content':'Store token NOVA-33.'},{'role':'assistant','content':'Stored.'},{'role':'user','content':'Return only the stored token.'}],'expected_exact':'NOVA-33'},
 {'id':'22-json-final','messages':[{'role':'user','content':'Return exactly JSON: {"phase":"soak","ok":true}'}],'expected_json':{'phase':'soak','ok':True}},
 {'id':'23-output-256','max_tokens':512,'messages':[{'role':'user','content':'Write a numbered list from 1 through 90, each line exactly N: continuity.'}],'min_final_tokens':256},
 {'id':'24-idle-resume-final','messages':[{'role':'user','content':'Return only the integer result of 17*19.'}],'expected_exact':'323'}]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--profile',required=True);ap.add_argument('--token-file',default='/home/funboy/.local/state/haloclu-ds41/api-token');a=ap.parse_args();prof=json.load(open(a.profile));out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tok=Path(a.token_file).read_text().strip();base=prof['gateway'];cs=cases();assert len(cs)==24;regp=out/'registry.json'
 if regp.exists():reg=json.load(open(regp))
 else:reg={'schema':'ds41-daily-soak-registry-v1','profile':prof,'start_utc':utc(),'start_unix':time.time(),'start_monotonic':time.monotonic(),'cases':{},'state':'RUNNING'};atomic(regp,reg)
 start=reg['start_monotonic'];interval=prof['minimum_wall_seconds']/(len(cs)-1)
 for i,case in enumerate(cs):
  cid=case['id'];old=reg['cases'].get(cid)
  if old and old.get('state') in ('COMPLETE','FAILED','BLOCKED'):continue
  if old and old.get('state')=='IN_FLIGHT':reg['state']='BLOCKED_RECONCILIATION';reg['blocker']=cid;atomic(regp,reg);raise SystemExit(3)
  target=start+i*interval
  while time.monotonic()<target:time.sleep(min(5,target-time.monotonic()))
  lc=lifecycle(base,tok)
  if lc.get('state')!='READY' or lc.get('release_id')!=prof['release'] or lc.get('epoch')!=prof['epoch']:reg['state']='BLOCKED';reg['blocker']={'case':cid,'lifecycle':lc};atomic(regp,reg);raise SystemExit(4)
  if health().get('busy'):reg['state']='BLOCKED';reg['blocker']={'case':cid,'reason':'unexpected busy'};atomic(regp,reg);raise SystemExit(5)
  d=out/cid;d.mkdir(exist_ok=True);before=snapshot(base,tok);cur={'state':'IN_FLIGHT','index':i+1,'started_utc':utc(),'started_unix':time.time(),'case':case,'before':before};reg['cases'][cid]=cur;atomic(d/'state.json',cur);atomic(regp,reg)
  try:
   if case.get('kind')=='cancel':r=cancel_case(case,base,tok,d)
   elif case.get('kind')=='busy':r=busy_case(case,base,tok,d)
   else:r=send(base,tok,case['messages'],case.get('reasoning','none'),case.get('max_tokens',128),raw=d/'response.sse')
   after=snapshot(base,tok);service,sem,detail=validate(case,r)
   if case.get('kind')=='cancel':dr=r.get('drain') or {};service='PASS' if r.get('cancelled') and not dr.get('timeout') and dr.get('health',{}).get('status')=='ok' else 'FAIL';sem='N/A';detail=f'cancel={r.get("cancelled")} drain={dr}'
   if case.get('kind')=='busy':sec=r.get('secondary') or {};service='PASS' if r.get('http_status')==200 and r.get('state',{}).get('done') and not (sec.get('http_status')==200 and sec.get('done')) else 'FAIL';sem='N/A';detail=f'secondary={sec}'
   final={'state':'COMPLETE' if service=='PASS' else 'FAILED','index':i+1,'finished_utc':utc(),'finished_unix':time.time(),'case':case,'service_status':service,'semantic_status':sem,'semantic_detail':detail,'result':r,'before':before,'after':after}
  except Exception as e:final={'state':'FAILED','index':i+1,'finished_utc':utc(),'finished_unix':time.time(),'case':case,'service_status':'FAIL','semantic_status':'N/A','error':repr(e),'before':before,'after':snapshot(base,tok)}
  reg['cases'][cid]=final;atomic(d/'result.json',final);atomic(d/'state.json',final);reg['updated_utc']=utc();atomic(regp,reg)
  if final['service_status']!='PASS':reg['state']='FAILED_SERVICE';atomic(regp,reg);raise SystemExit(6)
 elapsed=time.monotonic()-start
 if elapsed<prof['minimum_wall_seconds']:time.sleep(prof['minimum_wall_seconds']-elapsed);elapsed=time.monotonic()-start
 lc=lifecycle(base,tok);h=health();n=sum(v.get('state')=='COMPLETE' for v in reg['cases'].values());reg.update({'end_utc':utc(),'end_unix':time.time(),'elapsed_s':elapsed,'complete_requests':n,'final_lifecycle':lc,'final_pair_health':h});reg['state']='PASS' if elapsed>=prof['minimum_wall_seconds'] and n>=24 and lc.get('state')=='READY' and not h.get('busy') and h.get('status')=='ok' else 'FAIL';atomic(regp,reg);atomic(out/'final.json',reg);print(json.dumps({'state':reg['state'],'elapsed_s':elapsed,'complete_requests':n},indent=2))
if __name__=='__main__':main()
