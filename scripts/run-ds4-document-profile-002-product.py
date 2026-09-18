#!/usr/bin/env python3
from __future__ import annotations
import http.client, json, os, subprocess, time
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-document-profile-002'
OUT=BASE/'product'
CFG=ROOT/'runtime/ds41/config.ds4-document-profile-002.json'
BIN=Path('/home/funboy/.local/share/haloclu-ds41/current/bin/strixglm')
STATE=Path('/home/funboy/.local/state/ds4-document-profile-002')
OWNER='DS4_DOCUMENT_PROFILE_002_20260918'
MANIFEST=json.load(open(ROOT/'runtime/ds41/document-profile-002/prompt-manifest.json'))

def atomic(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def sh(cmd,check=True):
    return subprocess.run(cmd,text=True,capture_output=True,check=check)
def wait_http(url,code=200,timeout=120,headers=None):
    import urllib.request, urllib.error
    deadline=time.time()+timeout; last=None
    while time.time()<deadline:
        try:
            req=urllib.request.Request(url,headers=headers or {})
            with urllib.request.urlopen(req,timeout=3) as r:
                if r.status==code:return r.read()
                last=f'HTTP{r.status}'
        except Exception as e:last=repr(e)
        time.sleep(1)
    raise RuntimeError(f'timeout {url}: {last}')
def req(method,path,body=None,token=None,timeout=1800):
    c=http.client.HTTPConnection('127.0.0.1',18224,timeout=timeout)
    b=None if body is None else json.dumps(body,separators=(',',':')).encode()
    h={}
    if b is not None:h['Content-Type']='application/json'
    if token:h['Authorization']='Bearer '+token
    c.request(method,path,body=b,headers=h);r=c.getresponse();raw=r.read();status=r.status;ct=r.getheader('Content-Type') or '';c.close()
    try:o=json.loads(raw)
    except:o={'_raw':raw.decode(errors='replace')}
    return status,o,ct
def stream_once(body,token,cancel_after_first=False):
    c=http.client.HTTPConnection('127.0.0.1',18224,timeout=1800)
    h={'Content-Type':'application/json','Authorization':'Bearer '+token}
    c.request('POST','/v1/chat/completions',body=json.dumps(body,separators=(',',':')).encode(),headers=h)
    r=c.getresponse();out={'http':r.status,'reasoning':'','content':'','finish':None,'events':0,'cancelled':False}
    if r.status!=200: out['error']=r.read().decode(errors='replace');c.close();return out
    frame=[]
    while True:
        line=r.readline()
        if not line:break
        s=line.decode(errors='replace').rstrip('\r\n')
        if s=='':
            data=[x[5:].lstrip() for x in frame if x.startswith('data:')]
            frame=[]
            if not data:continue
            t='\n'.join(data)
            if t=='[DONE]':break
            try:o=json.loads(t)
            except:continue
            out['events']+=1
            for ch in o.get('choices') or []:
                d=ch.get('delta') or {}
                out['reasoning']+=d.get('reasoning_content') or d.get('reasoning') or ''
                out['content']+=d.get('content') or ''
                if ch.get('finish_reason') is not None:out['finish']=ch['finish_reason']
            if cancel_after_first and out['events']>=1:
                out['cancelled']=True;c.close();return out
        else:frame.append(s)
    c.close();return out
def lifecycle(token,action=None):
    if action:
        s,o,_=req('POST',f'/v1/lifecycle/{action}',{'confirm':True},token,30)
    else:s,o,_=req('GET','/v1/lifecycle',None,token,30)
    return s,o
def wait_lifecycle(token,want,timeout=1300):
    end=time.time()+timeout;last=None
    while time.time()<end:
        s,o=lifecycle(token);last=(s,o)
        if s==200 and o.get('state')==want:return o
        time.sleep(3)
    raise RuntimeError(f'lifecycle timeout {want}: {last}')
def exact_chat(token,text,expected,profile='document-low'):
    body={'model':'deepseek-v4.1-flash','profile':profile,'messages':[{'role':'user','content':text}],'temperature':0,'seed':1,'max_tokens':128,'stream':False}
    s,o,_=req('POST','/v1/chat/completions',body,token)
    content=((o.get('choices') or [{}])[0].get('message') or {}).get('content','') if isinstance(o,dict) else ''
    return {'http':s,'content':content,'expected':expected,'pass':s==200 and content.strip()==expected}
def document_chat(token,record):
    text=(ROOT/record['prompt_file']).read_text()
    body={'model':'deepseek-v4.1-flash','profile':'document-low','messages':[{'role':'user','content':text}],'temperature':0,'seed':1,'max_tokens':2048,'stream':False}
    t=time.monotonic();s,o,_=req('POST','/v1/chat/completions',body,token);wall=time.monotonic()-t
    msg=((o.get('choices') or [{}])[0].get('message') or {}) if isinstance(o,dict) else {}
    content=msg.get('content') or ''
    try: actual=json.loads(content); ok=(s==200 and actual==record['expected'])
    except Exception: actual=None; ok=False
    return {'http':s,'pass':ok,'expected':record['expected'],'actual':actual,'wall_s':wall,'reasoning_chars':len(msg.get('reasoning') or msg.get('reasoning_content') or ''),'final_chars':len(content),'finish_reason':((o.get('choices') or [{}])[0].get('finish_reason') if isinstance(o,dict) else None),'usage':o.get('usage') if isinstance(o,dict) else None}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'terminal.json').exists():raise SystemExit('replay guard')
    qt=json.load(open(BASE/'quality/terminal.json'))
    ct=json.load(open(BASE/'continuation/terminal.json'))
    scope=json.load(open(ROOT/'runtime/ds41/results/ds4-document-profile-002-scope-decision.json'))
    if not qt.get('six_pass') or not scope.get('product_admitted'):raise SystemExit('product gate not admitted for verified scope')
    STATE.mkdir(parents=True,exist_ok=True);os.chmod(STATE,0o700)
    # Start corrected CPU tokenizer.
    sh(['systemctl','--user','reset-failed','ds4-document-tokenizer.service'],check=False)
    sh(['systemd-run','--user','--unit=ds4-document-tokenizer','--collect','--property=Restart=no',
        f'--property=StandardOutput=append:{BASE}/tokenizer.log',f'--property=StandardError=append:{BASE}/tokenizer.log',
        str(ROOT/'scripts/run-ds4-usable-tokenizer.sh')])
    wait_http('http://127.0.0.1:18223/health',200,120)
    # Start protected candidate gateway.
    sh(['systemctl','--user','reset-failed','ds4-document-gateway.service'],check=False)
    sh(['systemd-run','--user','--unit=ds4-document-gateway','--collect','--property=Restart=no',
        f'--property=StandardOutput=append:{BASE}/gateway.log',f'--property=StandardError=append:{BASE}/gateway.log',
        str(BIN),'serve','--config',str(CFG)])
    wait_http('http://127.0.0.1:18224/health',200,120)
    token=(STATE/'api-token').read_text().strip()
    rows=[]
    # Auth gate.
    s,o,_=req('POST','/v1/chat/completions',{'messages':[{'role':'user','content':'x'}]},None,30)
    rows.append({'gate':'unauthorized_chat','pass':s==401,'http':s})
    s,l=lifecycle(token);rows.append({'gate':'lifecycle_ready_auth','pass':s==200 and l.get('state')=='READY','http':s,'body':l})
    code=next(x for x in MANIFEST['documents'] if x['id']=='code2k-middle-explicit-v2')
    docs=next(x for x in MANIFEST['documents'] if x['id']=='docs2k-middle-explicit-v2')
    dx=document_chat(token,code);rows.append({'gate':'gateway_document_code2k','pass':dx['pass'],'detail':dx})
    dy=document_chat(token,docs);rows.append({'gate':'gateway_document_docs2k','pass':dy['pass'],'detail':dy})
    x=exact_chat(token,'Compute 23+19. Return only the integer.','42');rows.append({'gate':'nonstream_low','pass':x['pass'],'detail':x})
    sb={'model':'deepseek-v4.1-flash','profile':'document-low','messages':[{'role':'user','content':'Compute 31+11. Return only the integer.'}],'temperature':0,'seed':1,'max_tokens':128,'stream':True}
    sx=stream_once(sb,token);rows.append({'gate':'sse_reasoning_final','pass':sx['http']==200 and sx['content'].strip()=='42' and sx['finish']=='stop','detail':sx})
    # Cancel one live stream, then prove a fresh request succeeds.
    cb={'model':'deepseek-v4.1-flash','profile':'document-low','messages':[{'role':'user','content':'List the integers from 1 through 500, one per line.'}],'temperature':0,'seed':1,'max_tokens':2048,'stream':True}
    cx=stream_once(cb,token,cancel_after_first=True)
    resume=None
    end=time.time()+180
    while time.time()<end:
        resume=exact_chat(token,'Return exactly DRAIN-OK.','DRAIN-OK')
        if resume['pass']:break
        time.sleep(3)
    rows.append({'gate':'cancel_drain_resume','pass':cx.get('cancelled') and bool(resume and resume['pass']),'cancel':cx,'resume':resume})
    # Whole-pair OFF, refusal without autoload, then ON and new request.
    so,bo=lifecycle(token,'off'); off=wait_lifecycle(token,'OFF',120)
    offchat=exact_chat(token,'Return exactly SHOULD-NOT-RUN.','SHOULD-NOT-RUN')
    controller=json.loads(sh([str(ROOT/'scripts/ds4-document-controller.sh'),'status']).stdout)
    rows.append({'gate':'off_no_autoload','pass':so==202 and offchat['http']==409 and controller.get('state')=='OFF','off_status':off,'chat':offchat,'controller':controller})
    sn,bn=lifecycle(token,'on'); on=wait_lifecycle(token,'READY',1300)
    post=exact_chat(token,'Return exactly ON-OK.','ON-OK')
    rows.append({'gate':'on_whole_pair_and_resume','pass':sn==202 and post['pass'],'on_status':on,'chat':post})
    ok=all(x['pass'] for x in rows)
    term={'schema':'ds4-document-profile-002-product-v1','owner':OWNER,'status':'PASS' if ok else 'FAIL','gates':rows,'finished_unix':time.time()}
    atomic(OUT/'terminal.json',term);print(json.dumps(term,ensure_ascii=False),flush=True)
    if not ok: subprocess.run([str(ROOT/'scripts/finalize-ds4-document-profile-002.sh'),'NOT_QUALIFIED'],check=True)
if __name__=='__main__':main()
