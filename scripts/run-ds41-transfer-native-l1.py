#!/usr/bin/env python3
from __future__ import annotations
import http.client, json, os, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/DS41-Q2-001/transfer-ds4-native-001/l1'
MAN=json.load(open(ROOT/'runtime/ds41/document-profile-002/prompt-manifest.json'))
HOST='127.0.0.1';PORT=18221
MODEL='DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2'

def atomic(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)

def rec(ident):
    for group in ('documents','holdouts'):
        for x in MAN[group]:
            if x['id']==ident:return x
    raise KeyError(ident)

def call(ident,record,independent_required=False):
    d=OUT/ident;d.mkdir(parents=True,exist_ok=True)
    sp=d/'state.json'
    if sp.exists():raise RuntimeError(f'replay guard {ident}')
    payload={
      'model':MODEL,
      'messages':[{'role':'user','content':(ROOT/record['prompt_file']).read_text()}],
      'temperature':0,'seed':1,'max_tokens':2048,'stream':False,
      'reasoning_effort':'low',
      'chat_template_kwargs':{
        'enable_thinking':True,
        'reasoning_effort':'low',
        'ds41_prompt_profile':'ds4-low-v1',
      },
    }
    atomic(sp,{'id':ident,'state':'IN_FLIGHT','started_unix':time.time(),'payload':payload,'expected':record['expected']})
    body=json.dumps(payload,separators=(',',':')).encode()
    t=time.monotonic()
    try:
        c=http.client.HTTPConnection(HOST,PORT,timeout=1800)
        c.request('POST','/v1/chat/completions',body=body,headers={'Content-Type':'application/json'})
        r=c.getresponse();raw=r.read();status=r.status;c.close()
    except Exception as e:
        st=json.load(open(sp));st.update(state='FAILED_TRANSPORT',finished_unix=time.time(),error=repr(e));atomic(sp,st)
        return {'id':ident,'status':'FAILED_TRANSPORT','error':repr(e)}
    wall=time.monotonic()-t
    try:o=json.loads(raw)
    except Exception as e:o={'_raw':raw.decode(errors='replace'),'_json_error':repr(e)}
    atomic(d/'response.json',o)
    msg=((o.get('choices') or [{}])[0].get('message') or {}) if isinstance(o,dict) else {}
    content=msg.get('content') or ''
    reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
    finish=((o.get('choices') or [{}])[0].get('finish_reason')) if isinstance(o,dict) else None
    try:actual=json.loads(content);semantic=(status==200 and actual==record['expected'])
    except:actual=None;semantic=False
    usage=o.get('usage') or {} if isinstance(o,dict) else {}
    ptd=usage.get('prompt_tokens_details') or {}
    metrics=o.get('metrics') if isinstance(o,dict) else None
    cached=ptd.get('cached_tokens')
    independent=(cached in (0,None))  # prefix cache is disabled in this release; None remains visible in raw.
    result={
      'id':ident,'status':'PASS' if semantic and (independent or not independent_required) else ('FAIL_INDEPENDENCE' if semantic else 'FAIL_SEMANTIC'),
      'semantic_pass':semantic,'independent':independent,'independent_required':independent_required,
      'http':status,'wall_s':wall,'expected':record['expected'],'actual':actual,
      'reasoning_chars':len(reasoning),'final_chars':len(content),'finish_reason':finish,
      'usage':usage,'metrics':metrics,
    }
    atomic(d/'validation.json',result)
    st=json.load(open(sp));st.update(state='COMPLETE',finished_unix=time.time(),result_status=result['status'],wall_s=wall);atomic(sp,st)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'terminal.json').exists():raise SystemExit('terminal exists')
    rows=[]
    def add(x):
        rows.append(x);atomic(OUT/'registry.json',{'schema':'ds41-transfer-native-l1-v1','requests_used':len(rows),'max_requests':6,'cases':rows,'updated_unix':time.time()});print(json.dumps(x,ensure_ascii=False),flush=True)
    add(call('native-code2k-v2',rec('code2k-middle-explicit-v2')))
    add(call('native-docs2k-v2',rec('docs2k-middle-explicit-v2')))
    originals=all(x['status']=='PASS' for x in rows[:2])
    if originals:
        add(call('native-holdout-a',rec('document-holdout-a')))
        add(call('native-holdout-b',rec('document-holdout-b')))
    holdouts=originals and len(rows)==4 and all(x['status']=='PASS' for x in rows[2:4])
    if holdouts:
        add(call('native-code2k-confirm',rec('code2k-middle-explicit-v2'),True))
        add(call('native-docs2k-confirm',rec('docs2k-middle-explicit-v2'),True))
    six=len(rows)==6 and all(x['status']=='PASS' for x in rows)
    term={'schema':'ds41-transfer-native-l1-terminal-v1','status':'L1_PASS' if six else 'L1_FAIL','requests_used':len(rows),'six_pass':six,'cases':rows,'finished_unix':time.time()}
    atomic(OUT/'terminal.json',term);print(json.dumps(term,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
