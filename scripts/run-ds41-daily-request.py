#!/usr/bin/env python3
from __future__ import annotations
import argparse, http.client, json, os, time
from pathlib import Path
from urllib.parse import urlparse

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n'); os.replace(tmp,path)
def feed_frame(lines, state, now, start):
    if not lines: return
    event='message'; data=[]
    for line in lines:
        if line.startswith('event:'): event=line[6:].strip()
        elif line.startswith('data:'): data.append(line[5:].lstrip())
    if not data: return
    text='\n'.join(data)
    if event=='haloclu.timing':
        try: state['gateway_timing_events'].append(json.loads(text))
        except: pass
        return
    if text=='[DONE]': state['done']=True; return
    try: obj=json.loads(text)
    except Exception as e: raise RuntimeError(f'malformed SSE JSON: {e}: {text[:200]}')
    if event=='error' or obj.get('error') is not None: raise RuntimeError(f'SSE error: {obj.get("error",obj)}')
    if obj.get('usage') is not None: state['usage']=obj['usage']
    if obj.get('metrics') is not None: state['metrics']=obj['metrics']
    for c in obj.get('choices') or []:
        d=c.get('delta') or {}
        reasoning=d.get('reasoning') or d.get('reasoning_content') or ''
        content=d.get('content') or ''
        if (reasoning or content) and state['first_any_s'] is None: state['first_any_s']=now-start
        if content and state['first_final_s'] is None: state['first_final_s']=now-start
        state['reasoning'] += reasoning; state['content'] += content
        if c.get('finish_reason') is not None: state['finish_reason']=c['finish_reason']; state['stop_reason']=c.get('stop_reason')
def derive(state, wall):
    u=state.get('usage') or {}; m=state.get('metrics') or {}; pt=u.get('prompt_tokens'); ct=u.get('completion_tokens')
    gen=m.get('generation_time_ms'); pair_ms=m.get('pair_prefill_engine_ms_max') or m.get('prefill_engine_ms'); computed=m.get('prompt_tokens_computed')
    return {
      'prompt_tokens':pt,'completion_tokens':ct,'reasoning_tokens':((u.get('completion_tokens_details') or {}).get('reasoning_tokens')),
      'prompt_tokens_computed':computed,'prompt_tokens_cached':m.get('prompt_tokens_cached'),'prompt_tokens_local_cache':m.get('prompt_tokens_local_cache'),'prompt_tokens_external_cache':m.get('prompt_tokens_external_cache'),'prompt_tokens_cache_creation':m.get('prompt_tokens_cache_creation'),
      'pair_prefill_engine_ms_max':m.get('pair_prefill_engine_ms_max'),'prefill_engine_ms_rank0':m.get('prefill_engine_ms'),
      'prefill_engine_tok_s_critical': (computed*1000/pair_ms if isinstance(computed,(int,float)) and isinstance(pair_ms,(int,float)) and pair_ms>0 else None),
      'engine_time_to_first_token_ms':m.get('time_to_first_token_ms'),'engine_queue_time_ms':m.get('queue_time_ms'),
      'client_ttft_any_s':state.get('first_any_s'),'client_first_final_s':state.get('first_final_s'),
      'input_over_client_ttft_tok_s':(pt/state['first_any_s'] if isinstance(pt,(int,float)) and state.get('first_any_s') else None),
      'decode_tps_engine':((ct-1)*1000/gen if isinstance(ct,(int,float)) and ct>1 and isinstance(gen,(int,float)) and gen>0 else None),
      'output_wall_tok_s':(ct/wall if isinstance(ct,(int,float)) and wall>0 else None),
      'draft_acceptance':((m.get('speculative_decoding') or {}).get('draft_acceptance_rate')),
      'speculative_decoding':m.get('speculative_decoding')
    }
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:18222'); ap.add_argument('--token-file',default='/home/funboy/.local/state/haloclu-ds41/api-token')
    ap.add_argument('--content-file'); ap.add_argument('--message'); ap.add_argument('--out'); ap.add_argument('--raw-sse')
    ap.add_argument('--label',required=False,default='request'); ap.add_argument('--reasoning',default='none'); ap.add_argument('--max-tokens',type=int,default=128); ap.add_argument('--context-tokens',type=int,default=65536); ap.add_argument('--timeout',type=int,default=1800); ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:
        s={'gateway_timing_events':[],'done':False,'first_any_s':None,'first_final_s':None,'reasoning':'','content':'','usage':None,'metrics':None,'finish_reason':None,'stop_reason':None}; st=100.0
        feed_frame([': heartbeat'],s,100.1,st); assert s['first_any_s'] is None
        feed_frame(['data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}'],s,100.2,st); assert s['first_any_s'] is None
        feed_frame(['data: {"choices":[{"index":0,"delta":{"reasoning":"r"},"finish_reason":null}],"usage":{"prompt_tokens":10,"completion_tokens":1}}'],s,100.3,st); assert abs(s['first_any_s']-.3)<1e-6 and s['first_final_s'] is None
        feed_frame(['data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}],"metrics":{"prefill_engine_ms":20,"pair_prefill_engine_ms_max":25,"prompt_tokens_computed":10,"generation_time_ms":5}}'],s,100.4,st); assert abs(s['first_final_s']-.4)<1e-6
        feed_frame(['data: [DONE]'],s,100.5,st); assert s['done']; print('SELF_TEST=PASS'); return
    content=Path(a.content_file).read_text() if a.content_file else a.message
    if content is None or not a.out: raise SystemExit('content/message and --out required')
    token=Path(a.token_file).read_text().strip(); u=urlparse(a.url); conn=http.client.HTTPConnection(u.hostname,u.port,timeout=a.timeout)
    payload={'model':'DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2','messages':[{'role':'user','content':content}],'temperature':0,'seed':1,'stream':True,'stream_options':{'include_usage':True,'continuous_usage_stats':True},'max_tokens':a.max_tokens,'context_tokens':a.context_tokens,'reasoning_effort':a.reasoning,'chat_template_kwargs':{'reasoning_effort':a.reasoning}}
    body=json.dumps(payload,separators=(',',':')).encode(); start=time.monotonic(); conn.request('POST','/v1/chat/completions',body=body,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Accept':'text/event-stream','X-HaloClu-Timings':'1'})
    resp=conn.getresponse(); headers_s=time.monotonic()-start
    if resp.status!=200: data=resp.read(8192).decode(errors='replace'); raise SystemExit(f'HTTP {resp.status}: {data}')
    state={'gateway_timing_events':[],'done':False,'first_any_s':None,'first_final_s':None,'reasoning':'','content':'','usage':None,'metrics':None,'finish_reason':None,'stop_reason':None}
    raw=[]; frame=[]
    while True:
        b=resp.readline()
        if not b: break
        line=b.decode('utf-8',errors='strict').rstrip('\r\n'); raw.append(line)
        if line=='':
            feed_frame(frame,state,time.monotonic(),start); frame=[]
            if state['done']: break
        else: frame.append(line)
    if frame: feed_frame(frame,state,time.monotonic(),start)
    wall=time.monotonic()-start; conn.close()
    result={'schema':'ds41-daily-request-v1','label':a.label,'url':a.url,'settings':{'reasoning':a.reasoning,'max_tokens':a.max_tokens,'context_tokens':a.context_tokens,'stream':True},'http':{'status':resp.status,'headers_s':headers_s,'wall_s':wall},'state':state,'derived':derive(state,wall)}
    if not state['done']: result['status']='INCOMPLETE'; result['error']='SSE ended without DONE'
    elif state['finish_reason'] is None: result['status']='INCOMPLETE'; result['error']='missing finish_reason'
    else: result['status']='PASS'
    atomic_json(Path(a.out),result)
    if a.raw_sse: Path(a.raw_sse).write_text('\n'.join(raw)+'\n')
    print(json.dumps({'status':result['status'],'label':a.label,'http':result['http'],'finish_reason':state['finish_reason'],'content_chars':len(state['content']),'reasoning_chars':len(state['reasoning']),'derived':result['derived']},ensure_ascii=False))
if __name__=='__main__': main()
