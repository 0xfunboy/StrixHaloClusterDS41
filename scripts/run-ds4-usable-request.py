#!/usr/bin/env python3
from __future__ import annotations
import argparse, http.client, json, os, re, socket, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

def atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    os.replace(tmp,path)

def unit_pid(unit:str, peer=False):
    cmd=['systemctl','--user','show',unit,'-p','MainPID','--value']
    if peer: cmd=['ssh','-o','IdentityAgent=none','-o','BatchMode=yes','02-evo-x3-tb']+cmd
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=8)
    try: return int(p.stdout.strip())
    except: return 0

def proc_snapshot(pid:int, peer=False):
    if pid<=0: return {'pid':pid,'available':False}
    if peer:
        script=f"cat /proc/{pid}/status; echo __STAT__; cat /proc/{pid}/stat"
        p=subprocess.run(['ssh','-o','IdentityAgent=none','-o','BatchMode=yes','02-evo-x3-tb','bash','-lc',script],
                         text=True,capture_output=True,timeout=8)
        text=p.stdout
    else:
        try: text=Path(f'/proc/{pid}/status').read_text()+'\n__STAT__\n'+Path(f'/proc/{pid}/stat').read_text()
        except Exception as e: return {'pid':pid,'available':False,'error':repr(e)}
    a,b=(text.split('__STAT__',1)+[''])[:2]
    vals={}
    for line in a.splitlines():
        if ':' in line:
            k,v=line.split(':',1); vals[k]=v.strip()
    stat=b.strip()
    after=stat[stat.rfind(')')+2:].split() if ')' in stat else []
    # after begins with field 3 (state): indexes 7=minflt(field10), 9=majflt(field12)
    def num(i):
        try:return int(after[i])
        except:return None
    return {'pid':pid,'available':True,'VmRSS':vals.get('VmRSS'),'VmHWM':vals.get('VmHWM'),
            'VmSwap':vals.get('VmSwap'),'minflt':num(7),'majflt':num(9)}

def resources():
    lp=unit_pid('ds4-usable-coordinator.service',False)
    rp=unit_pid('ds4-usable-worker.service',True)
    return {'unix':time.time(),'node01':proc_snapshot(lp,False),'node02':proc_snapshot(rp,True)}

def feed_frame(lines,state,now,start):
    if not lines:return
    event='message'; data=[]
    for line in lines:
        if line.startswith('event:'):event=line[6:].strip()
        elif line.startswith('data:'):data.append(line[5:].lstrip())
    if not data:return
    text='\n'.join(data)
    if text=='[DONE]':state['done']=True;return
    try: obj=json.loads(text)
    except Exception as e: raise RuntimeError(f'malformed SSE JSON: {e}: {text[:300]}')
    if event=='error' or obj.get('error') is not None: raise RuntimeError(f'SSE error: {obj}')
    if obj.get('usage') is not None:state['usage']=obj['usage']
    if obj.get('metrics') is not None:state['metrics']=obj['metrics']
    for c in obj.get('choices') or []:
        d=c.get('delta') or c.get('message') or {}
        reasoning=d.get('reasoning_content') or d.get('reasoning') or ''
        content=d.get('content') or ''
        if (reasoning or content) and state['first_any_s'] is None:state['first_any_s']=now-start
        if reasoning and state['first_reasoning_s'] is None:state['first_reasoning_s']=now-start
        if content and state['first_final_s'] is None:state['first_final_s']=now-start
        state['reasoning']+=reasoning;state['content']+=content
        if c.get('finish_reason') is not None:state['finish_reason']=c['finish_reason']

def parse_server_log(seg:str):
    out={'raw_bytes':len(seg.encode()),'cache_lines':[]}
    for line in seg.splitlines():
        if 'live kv ' in line or 'cached=' in line and 'ds4-server:' in line: out['cache_lines'].append(line)
        m=re.search(r'prefill chunk\s+(\d+)/(\d+).*?avg=([0-9.]+) t/s\s+([0-9.]+)s',line)
        if m: out['prefill']={'done_tokens':int(m.group(1)),'prompt_tokens':int(m.group(2)),'avg_tps':float(m.group(3)),'engine_s':float(m.group(4))}
        m=re.search(r'prompt done\s+([0-9.]+)s',line)
        if m:out['prompt_done_s']=float(m.group(1))
        m=re.search(r'gen=(\d+).*?decoding chunk=([0-9.]+) t/s avg=([0-9.]+) t/s\s+([0-9.]+)s',line)
        if m:out['decode']={'generated':int(m.group(1)),'last_chunk_tps':float(m.group(2)),'avg_tps':float(m.group(3)),'engine_decode_s':float(m.group(4))}
        m=re.search(r'gen=(\d+)\s+finish=([^\s]+)\s+([0-9.]+)s',line)
        if m:out['finish']={'generated':int(m.group(1)),'reason':m.group(2),'engine_total_s':float(m.group(3))}
    return out

def do_request(url,payload,timeout,cancel_after=None,server_log=None):
    u=urlparse(url); conn=http.client.HTTPConnection(u.hostname,u.port,timeout=timeout)
    body=json.dumps(payload,separators=(',',':')).encode()
    log_start=Path(server_log).stat().st_size if server_log and Path(server_log).exists() else 0
    before=resources() if not payload.get('_selftest') else None
    payload.pop('_selftest',None); body=json.dumps(payload,separators=(',',':')).encode()
    start=time.monotonic()
    conn.request('POST',u.path or '/v1/chat/completions',body=body,headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    resp=conn.getresponse(); headers_s=time.monotonic()-start
    if resp.status!=200:
        data=resp.read(8192).decode(errors='replace'); conn.close(); raise RuntimeError(f'HTTP {resp.status}: {data}')
    state={'done':False,'first_any_s':None,'first_reasoning_s':None,'first_final_s':None,'reasoning':'','content':'','usage':None,'metrics':None,'finish_reason':None,'cancelled':False}
    raw=[]; frame=[]
    while True:
        if cancel_after is not None and time.monotonic()-start>=cancel_after:
            state['cancelled']=True; break
        b=resp.readline()
        if not b:break
        line=b.decode('utf-8',errors='strict').rstrip('\r\n');raw.append(line)
        if line=='':
            feed_frame(frame,state,time.monotonic(),start);frame=[]
            if state['done']:break
        else:frame.append(line)
    conn.close(); wall=time.monotonic()-start
    after=resources() if before is not None else None
    seg=''
    if server_log and Path(server_log).exists():
        with open(server_log,'rb') as f:f.seek(log_start);seg=f.read().decode(errors='replace')
    usage=state.get('usage') or {}; pd=usage.get('prompt_tokens_details') or {}
    result={'http_status':resp.status,'headers_s':headers_s,'wall_s':wall,'state':state,
            'cache':{'prompt_tokens':usage.get('prompt_tokens'),'completion_tokens':usage.get('completion_tokens'),
                     'cached_tokens':pd.get('cached_tokens'),'cache_write_tokens':pd.get('cache_write_tokens')},
            'server':parse_server_log(seg),'resources_before':before,'resources_after':after}
    return result,'\n'.join(raw)+'\n',seg

class Mock(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*a):pass
    def do_POST(self):
        if self.path=='/timeout':
            time.sleep(.3); return
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers()
        def send(x):
            try:self.wfile.write((x+'\n\n').encode());self.wfile.flush();return True
            except: return False
        if self.path=='/sse':
            send('data: {"choices":[{"delta":{"reasoning_content":"r"},"finish_reason":null}]}')
            time.sleep(.02)
            send('data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":3}}}')
            send('data: [DONE]')
        elif self.path=='/cancel':
            for i in range(100):
                if not send('data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}'):break
                time.sleep(.02)

def selftest():
    s=ThreadingHTTPServer(('127.0.0.1',0),Mock);threading.Thread(target=s.serve_forever,daemon=True).start();port=s.server_port
    p={'_selftest':True,'model':'x','messages':[{'role':'user','content':'x'}],'stream':True}
    r,_,_=do_request(f'http://127.0.0.1:{port}/sse',dict(p),2)
    assert r['state']['content']=='x' and r['state']['reasoning']=='r' and r['state']['done'] and r['state']['first_reasoning_s'] < r['state']['first_final_s']
    timed=False
    try: do_request(f'http://127.0.0.1:{port}/timeout',dict(p),.1)
    except (socket.timeout,TimeoutError,http.client.RemoteDisconnected):timed=True
    assert timed
    r,_,_=do_request(f'http://127.0.0.1:{port}/cancel',dict(p),2,cancel_after=.05)
    assert r['state']['cancelled'] and r['wall_s'] < 1
    s.shutdown();print('SELF_TEST=SSE_PASS TIMEOUT_PASS CANCEL_PASS')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--id');ap.add_argument('--out-dir');ap.add_argument('--prompt-file');ap.add_argument('--messages-json')
    ap.add_argument('--url',default='http://127.0.0.1:8080/v1/chat/completions');ap.add_argument('--max-tokens',type=int,default=128)
    ap.add_argument('--thinking',choices=['on','off'],default='off');ap.add_argument('--reasoning-effort',default='none');ap.add_argument('--timeout',type=int,default=1800)
    ap.add_argument('--server-log');ap.add_argument('--cancel-after',type=float)
    a=ap.parse_args()
    if a.self_test:return selftest()
    if not a.id or not a.out_dir:raise SystemExit('--id and --out-dir required')
    if bool(a.prompt_file)==bool(a.messages_json):raise SystemExit('exactly one of --prompt-file/--messages-json')
    messages=[{'role':'user','content':Path(a.prompt_file).read_text()}] if a.prompt_file else json.load(open(a.messages_json))
    payload={'model':'deepseek-v4.1-flash','messages':messages,'temperature':0,'seed':1,'max_tokens':a.max_tokens,'stream':True,
             'stream_options':{'include_usage':True},'thinking':a.thinking=='on','reasoning_effort':a.reasoning_effort,
             'chat_template_kwargs':{'enable_thinking':a.thinking=='on','reasoning_effort':a.reasoning_effort}}
    d=Path(a.out_dir)/a.id;d.mkdir(parents=True,exist_ok=True);sp=d/'state.json'
    if sp.exists():raise SystemExit(f'replay guard: {sp} exists')
    atomic(sp,{'id':a.id,'state':'IN_FLIGHT','started_unix':time.time(),'payload':payload,'prompt_file':a.prompt_file,'messages_json':a.messages_json})
    try:r,raw,seg=do_request(a.url,payload,a.timeout,a.cancel_after,a.server_log)
    except Exception as e:
        st=json.load(open(sp));st.update(state='FAILED_TRANSPORT',finished_unix=time.time(),error=repr(e));atomic(sp,st);raise
    (d/'raw.sse').write_text(raw);(d/'server.log.slice').write_text(seg);atomic(d/'result.json',r)
    st=json.load(open(sp));st.update(state='COMPLETE',finished_unix=time.time(),finish_reason=r['state']['finish_reason'],content_chars=len(r['state']['content']),reasoning_chars=len(r['state']['reasoning']),cancelled=r['state']['cancelled']);atomic(sp,st)
    print(json.dumps({'id':a.id,'state':st['state'],'wall_s':r['wall_s'],'first_final_s':r['state']['first_final_s'],'first_reasoning_s':r['state']['first_reasoning_s'],'cache':r['cache'],'server':r['server'],'finish_reason':r['state']['finish_reason'],'content':r['state']['content']},ensure_ascii=False))
if __name__=='__main__':main()
