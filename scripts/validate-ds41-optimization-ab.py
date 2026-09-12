#!/usr/bin/env python3
"""Post-run validator for DS41 same-load EP skip-remote A/B."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path


def result_map(doc): return {r['label']:r for r in doc.get('results',[])}

def mean(xs): return sum(xs)/len(xs) if xs else None

def load_code_validator():
    p=Path(__file__).with_name('validate-ds41-offline-suite.py')
    spec=importlib.util.spec_from_file_location('ds41_suite_validator',p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.validate_code

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank0',required=True); ap.add_argument('--rank1',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    d0=json.load(open(a.rank0)); d1=json.load(open(a.rank1)); r0=result_map(d0); r1=result_map(d1)
    labels=sorted(set(r0)|set(r1)); identity={lab:(lab in r0 and lab in r1 and r0[lab].get('token_ids')==r1[lab].get('token_ids')) for lab in labels}
    speed_labels=['ab-baseline-1','ab-baseline-2','ab-candidate-1','ab-candidate-2']
    same_workload=all(r0[x]['token_ids']==r0[speed_labels[0]]['token_ids'] for x in speed_labels if x in r0)
    def perf(prefix):
        rows=[]
        for i in (1,2):
            x=r0[f'ab-{prefix}-{i}']; rows.append({'label':x['label'],'completion_tokens':x['completion_token_count'],'tps':x['derived']['decode_tps_first_to_last'],'ttft_s':x['client']['ttft_s'],'wall_s':x['client']['wall_s']})
        return {'runs':rows,'decode_tps_mean':mean([x['tps'] for x in rows]),'ttft_s_mean':mean([x['ttft_s'] for x in rows]),'wall_s_mean':mean([x['wall_s'] for x in rows])}
    baseline=perf('baseline'); candidate=perf('candidate')
    gain=(candidate['decode_tps_mean']/baseline['decode_tps_mean']-1)*100
    smoke=r0['ab-candidate-smoke']; smoke_pass=smoke.get('finished') and smoke.get('text','').strip()=='323'
    coding=load_code_validator()(r0['ab-candidate-coding'].get('text',''))
    try:
        obj=json.loads(r0['ab-candidate-json'].get('text','')); json_pass=(obj=={'total':17,'valid_ids':['a','c']}); json_result={'status':'PASS' if json_pass else 'FAIL','parsed':obj}
    except Exception as e:
        json_pass=False; json_result={'status':'FAIL','error':f'{type(e).__name__}: {e}'}
    ok=(d0.get('status')=='OPTIMIZATION_AB_COMPLETE' and d1.get('status')=='OPTIMIZATION_AB_COMPLETE' and all(identity.values()) and same_workload and smoke_pass and coding.get('status')=='PASS' and json_pass)
    out={'status':'PASS' if ok else 'FAIL','rank_token_identity':identity,'speed_baseline_candidate_token_identity':same_workload,'baseline':baseline,'candidate':candidate,'decode_tps_gain_pct':gain,'smoke':{'pass':smoke_pass,'text':smoke.get('text'),'tokens':smoke.get('completion_token_count'),'ttft_s':smoke.get('client',{}).get('ttft_s'),'wall_s':smoke.get('client',{}).get('wall_s')},'coding':coding,'json':json_result}
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(out,indent=2,ensure_ascii=False)); raise SystemExit(0 if ok else 1)
if __name__=='__main__': main()
