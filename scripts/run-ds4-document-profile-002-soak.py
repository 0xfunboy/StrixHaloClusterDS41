#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, http.client, subprocess
from pathlib import Path

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
BASE=ROOT/'reports/DS41-Q2-001/ds4-document-profile-002'
OUT=BASE/'soak'
STATE=Path('/home/funboy/.local/state/ds4-document-profile-002')
OWNER='DS4_DOCUMENT_PROFILE_002_20260918'
MANIFEST=json.load(open(ROOT/'runtime/ds41/document-profile-002/prompt-manifest.json'))
AUD=MANIFEST['documents']
HOLDS=MANIFEST['holdouts']

def atomic(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)

def req(token,prompt,expected):
    c=http.client.HTTPConnection('127.0.0.1',18224,timeout=1800)
    body={'model':'deepseek-v4.1-flash','profile':'document-low','messages':[{'role':'user','content':prompt}],'temperature':0,'seed':1,'max_tokens':2048,'stream':False}
    t=time.monotonic();c.request('POST','/v1/chat/completions',body=json.dumps(body,separators=(',',':')).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
    r=c.getresponse();raw=r.read();wall=time.monotonic()-t;c.close()
    try:o=json.loads(raw)
    except:o={'_raw':raw.decode(errors='replace')}
    content=((o.get('choices') or [{}])[0].get('message') or {}).get('content','') if isinstance(o,dict) else ''
    try:got=json.loads(content);ok=(r.status==200 and got==expected)
    except:got=None;ok=False
    return {'http':r.status,'wall_s':wall,'pass':ok,'actual':got,'expected':expected,'finish_reason':((o.get('choices') or [{}])[0].get('finish_reason') if isinstance(o,dict) else None),'usage':o.get('usage') if isinstance(o,dict) else None}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'terminal.json').exists():raise SystemExit('replay guard')
    prod=json.load(open(BASE/'product/terminal.json'))
    if prod.get('status')!='PASS':raise SystemExit('product gate not PASS')
    token=(STATE/'api-token').read_text().strip()
    code=next(x for x in AUD if x['id']=='code2k-middle-explicit-v2')
    docs=next(x for x in AUD if x['id']=='docs2k-middle-explicit-v2')
    ha=next(x for x in HOLDS if x['id']=='document-holdout-a')
    hb=next(x for x in HOLDS if x['id']=='document-holdout-b')
    cases=[
      ('code2k',ROOT/code['prompt_file'],code['expected']),
      ('docs2k',ROOT/docs['prompt_file'],docs['expected']),
      ('holdout-a',ROOT/ha['prompt_file'],ha['expected']),
      ('holdout-b',ROOT/hb['prompt_file'],hb['expected']),
      ('code2k-confirm',ROOT/code['prompt_file'],code['expected']),
      ('docs2k-confirm',ROOT/docs['prompt_file'],docs['expected']),
    ]
    plan=(cases*4)[:24]
    start=time.time();rows=[]
    atomic(OUT/'registry.json',{'schema':'ds4-document-profile-002-soak-v1','owner':OWNER,'state':'IN_FLIGHT','started_unix':start,'required_seconds':7200,'required_requests':24,'cases':rows})
    for i,(name,p,exp) in enumerate(plan,1):
        row=req(token,p.read_text(),exp);row.update(index=i,case=name,finished_unix=time.time());rows.append(row)
        atomic(OUT/'registry.json',{'schema':'ds4-document-profile-002-soak-v1','owner':OWNER,'state':'IN_FLIGHT','started_unix':start,'required_seconds':7200,'required_requests':24,'cases':rows})
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if not row['pass']:
            term={'schema':'ds4-document-profile-002-soak-terminal-v1','owner':OWNER,'status':'FAIL','started_unix':start,'finished_unix':time.time(),'elapsed_s':time.time()-start,'cases':rows}
            atomic(OUT/'terminal.json',term)
            subprocess.run([str(ROOT/'scripts/finalize-ds4-document-profile-002.sh'),'NOT_QUALIFIED'],check=True)
            return
        if i<24: time.sleep(300)
    elapsed=time.time()-start
    if elapsed<7200: time.sleep(7200-elapsed)
    end=time.time()
    term={'schema':'ds4-document-profile-002-soak-terminal-v1','owner':OWNER,'status':'PASS','started_unix':start,'finished_unix':end,'elapsed_s':end-start,'cases':rows}
    atomic(OUT/'terminal.json',term)
    atomic(OUT/'registry.json',{'schema':'ds4-document-profile-002-soak-v1','owner':OWNER,'state':'COMPLETE','started_unix':start,'finished_unix':end,'required_seconds':7200,'required_requests':24,'cases':rows})
    subprocess.run([str(ROOT/'scripts/finalize-ds4-document-profile-002.sh'),'QUALIFIED'],check=True)
    print(json.dumps(term,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
