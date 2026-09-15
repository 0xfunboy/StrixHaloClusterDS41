#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=ROOT/'reports/DS41-Q2-001/daily-k2-run/panel'
MAN=ROOT/'reports/DS41-Q2-001/daily-k2-panel/corpora/manifest.json'
EXPECTED_RELEASE='c075e82464c954e202c6f47658821d233edc13a5'
EXPECTED_EPOCH='1789482113148506636'
TOKEN=Path('/home/funboy/.local/state/haloclu-ds41/api-token').read_text().strip()

def atomic(p:Path,obj):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n'); os.replace(q,p)
def get_lifecycle():
    req=urllib.request.Request('http://127.0.0.1:18222/v1/lifecycle',headers={'Authorization':'Bearer '+TOKEN})
    with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)
def check_live():
    x=get_lifecycle()
    if x.get('state')!='READY' or x.get('release_id')!=EXPECTED_RELEASE or x.get('epoch')!=EXPECTED_EPOCH:
        raise RuntimeError('live identity mismatch: '+json.dumps(x,sort_keys=True))

def run_one(kind,label,sample,max_tokens=128):
    ident=f'{kind}-{label}-s{sample}'
    d=OUT/ident; d.mkdir(parents=True,exist_ok=True)
    statep=d/'state.json'; result=d/'result.json'; validation=d/'validation.json'
    if statep.exists():
        st=json.load(open(statep))
        if st.get('state')=='COMPLETE': return st
        if st.get('state')=='IN_FLIGHT':
            # Never resend unknown in-flight work automatically.
            raise RuntimeError(f'{ident} remains IN_FLIGHT; reconcile before retry')
    check_live()
    st={'id':ident,'kind':kind,'label':label,'sample':sample,'state':'IN_FLIGHT','release':EXPECTED_RELEASE,'epoch':EXPECTED_EPOCH,'started_at':time.time(),'max_tokens':max_tokens,'context_tokens':65536}
    atomic(statep,st)
    corpus=ROOT/f'reports/DS41-Q2-001/daily-k2-panel/corpora/{kind}-{label}.txt'
    cmd=['python3',str(ROOT/'scripts/run-ds41-daily-request.py'),'--label',ident,'--content-file',str(corpus),'--reasoning','none','--max-tokens',str(max_tokens),'--context-tokens','65536','--out',str(result),'--raw-sse',str(d/'raw.sse')]
    p=subprocess.run(cmd,text=True,capture_output=True)
    (d/'collector.stdout').write_text(p.stdout); (d/'collector.stderr').write_text(p.stderr)
    st['collector_rc']=p.returncode
    if p.returncode!=0:
        st.update(state='FAILED',finished_at=time.time(),error=(p.stderr or p.stdout)[-4000:]); atomic(statep,st); return st
    vcmd=['python3',str(ROOT/'scripts/validate-ds41-daily-context.py'),'--result',str(result),'--manifest',str(MAN),'--kind',kind,'--label',label]
    v=subprocess.run(vcmd,text=True,capture_output=True)
    (d/'validator.stderr').write_text(v.stderr); validation.write_text(v.stdout)
    st['validator_rc']=v.returncode
    try: val=json.loads(v.stdout)
    except Exception: val={'status':'ERROR','raw':v.stdout[-2000:]}
    st['validation_status']=val.get('status'); st['state']='COMPLETE' if v.returncode==0 else 'FAILED'; st['finished_at']=time.time(); atomic(statep,st)
    # Atomic campaign checkpoint after every request.
    update_registry()
    return st

def update_registry():
    rows=[]
    if OUT.exists():
        for p in sorted(OUT.glob('*/state.json')):
            try: rows.append(json.load(open(p)))
            except Exception: pass
    atomic(OUT/'registry.json',{'schema':'ds41-daily-panel-registry-v1','release':EXPECTED_RELEASE,'epoch':EXPECTED_EPOCH,'updated_at':time.time(),'cases':rows})

def all_pass(kind,label,n):
    return all((lambda p: p.exists() and json.load(open(p)).get('state')=='COMPLETE')(OUT/f'{kind}-{label}-s{i}'/'state.json') for i in range(1,n+1))

def main():
    OUT.mkdir(parents=True,exist_ok=True); update_registry()
    seq=[('code','2k',1),('code','4k',1),('code','8k',3),('docs','8k',1),('code','16k',1),('code','32k',3),('docs','32k',1)]
    for kind,label,n in seq:
        for i in range(1,n+1):
            st=run_one(kind,label,i)
            print(json.dumps(st,sort_keys=True),flush=True)
            if st.get('state')!='COMPLETE':
                # Stop only dependent size escalation; leave independent branches to caller.
                print(f'DEPENDENT_STOP {kind}-{label}',flush=True); return 2
    # 64K conditional: both code32k and docs32k passed, and 62556+128 < public 65536.
    if all_pass('code','32k',3) and all_pass('docs','32k',1):
        for kind in ('code','docs'):
            st=run_one(kind,'64k',1); print(json.dumps(st,sort_keys=True),flush=True)
            if st.get('state')!='COMPLETE': return 3
    update_registry(); print('PANEL_COMPLETE',flush=True); return 0
if __name__=='__main__': raise SystemExit(main())
