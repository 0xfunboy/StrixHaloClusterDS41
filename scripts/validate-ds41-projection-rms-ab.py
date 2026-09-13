#!/usr/bin/env python3
"""Independent validator for DS41 same-load projection/RMS A/B (AB / BA / AB)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

HISTORICAL_TPS = 9.72199209460323


def result_map(doc):
    return {r["label"]: r for r in doc.get("results", [])}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_code_validator():
    p = Path(__file__).with_name("validate-ds41-offline-suite.py")
    spec = importlib.util.spec_from_file_location("ds41_suite_validator", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_code


def perf(rows, prefix):
    out=[]
    for i in (1,2,3):
        r=rows[f"projection-ab-{prefix}-{i}"]
        out.append({
            "label":r["label"],
            "completion_tokens":r["completion_token_count"],
            "finish_reason":r.get("finish_reason"),
            "decode_tps":r["derived"]["decode_tps_first_to_last"],
            "ttft_s":r["client"]["ttft_s"],
            "wall_s":r["client"]["wall_s"],
        })
    tps=[x["decode_tps"] for x in out]
    m=mean(tps)
    sd=statistics.stdev(tps) if len(tps)>1 else 0.0
    return {
        "runs":out,
        "decode_tps_mean":m,
        "decode_tps_sample_sd":sd,
        "decode_tps_cv_pct":sd/m*100.0 if m else None,
        "decode_tps_min":min(tps),
        "decode_tps_max":max(tps),
        "ttft_s_mean":mean([x["ttft_s"] for x in out]),
        "wall_s_mean":mean([x["wall_s"] for x in out]),
    }


def first_diff(a,b):
    n=min(len(a),len(b))
    for i in range(n):
        if a[i]!=b[i]: return i
    return None if len(a)==len(b) else n


def reasoning_final(text):
    return text.rsplit("</think>",1)[1].strip() if "</think>" in text else text.strip()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rank0',required=True)
    ap.add_argument('--rank1',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    d0=json.load(open(a.rank0)); d1=json.load(open(a.rank1))
    r0,r1=result_map(d0),result_map(d1)
    labels=sorted(set(r0)|set(r1))
    rank_identity={lab:(lab in r0 and lab in r1 and r0[lab].get('token_ids')==r1[lab].get('token_ids')) for lab in labels}

    baseline=perf(r0,'baseline'); candidate=perf(r0,'candidate')
    overall_gain=(candidate['decode_tps_mean']/baseline['decode_tps_mean']-1)*100.0
    wall_gain=(baseline['wall_s_mean']/candidate['wall_s_mean']-1)*100.0
    wall_reduction=(1-candidate['wall_s_mean']/baseline['wall_s_mean'])*100.0
    historical_gain=(candidate['decode_tps_mean']/HISTORICAL_TPS-1)*100.0
    paired=[]
    for i in (1,2,3):
        av=r0[f'projection-ab-baseline-{i}']['derived']['decode_tps_first_to_last']
        bv=r0[f'projection-ab-candidate-{i}']['derived']['decode_tps_first_to_last']
        paired.append({'pair':i,'a_tps':av,'b_tps':bv,'gain_pct':(bv/av-1)*100.0})

    ab_tokens=[]
    for i in (1,2,3):
        aa=r0[f'projection-ab-baseline-{i}']['token_ids']; bb=r0[f'projection-ab-candidate-{i}']['token_ids']
        ab_tokens.append({'pair':i,'identical':aa==bb,'first_diff_index':first_diff(aa,bb),'a_tokens':len(aa),'b_tokens':len(bb)})

    # Enforce the frozen speed protocol.
    speed_protocol_pass=all(
        r0[f'projection-ab-{arm}-{i}']['completion_token_count']==128
        and r0[f'projection-ab-{arm}-{i}'].get('finish_reason')=='length'
        for arm in ('baseline','candidate') for i in (1,2,3)
    )

    smoke=r0['projection-ab-candidate-smoke']; smoke_pass=smoke.get('finished') and smoke.get('text','').strip()=='323'
    coding=load_code_validator()(r0['projection-ab-candidate-coding'].get('text',''))
    try:
        obj=json.loads(r0['projection-ab-candidate-json'].get('text',''))
        json_pass=obj=={'total':17,'valid_ids':['a','c']}
        json_result={'status':'PASS' if json_pass else 'FAIL','parsed':obj}
    except Exception as exc:
        json_pass=False; json_result={'status':'FAIL','error':f'{type(exc).__name__}: {exc}'}
    reasoning=r0['projection-ab-candidate-reasoning-high']; ans=reasoning_final(reasoning.get('text',''))
    reasoning_pass=reasoning.get('finished') and ans=='10'

    s0=d0.get('projection_stats') or {}; s1=d1.get('projection_stats') or {}
    execution_pass=(
        s0.get('tilelang_calls',0)>0 and s1.get('tilelang_calls',0)>0
        and (s0.get('fallback_reasons') or {}).get('tokens_not_1',0)>0
        and (s1.get('fallback_reasons') or {}).get('tokens_not_1',0)>0
    )
    same_lib=(d0.get('native_hip_identity') or {}).get('sha256')==(d1.get('native_hip_identity') or {}).get('sha256')
    order_pass=d0.get('order')==['A1','B1','B2','A2','A3','B3'] and d1.get('order')==d0.get('order')

    ok=(
        d0.get('status')=='MHC_PROJECTION_AB_COMPLETE'
        and d1.get('status')=='MHC_PROJECTION_AB_COMPLETE'
        and all(rank_identity.values()) and speed_protocol_pass and order_pass
        and smoke_pass and coding.get('status')=='PASS' and json_pass and reasoning_pass
        and execution_pass and same_lib
    )
    out={
        'status':'PASS' if ok else 'FAIL',
        'historical_promoted_tps':HISTORICAL_TPS,
        'rank_token_identity':rank_identity,
        'speed_protocol_pass':speed_protocol_pass,
        'order_pass':order_pass,
        'baseline':baseline,
        'candidate':candidate,
        'paired_gains':paired,
        'decode_tps_gain_pct':overall_gain,
        'gain_vs_historical_pct':historical_gain,
        'offline_wall_throughput_gain_pct':wall_gain,
        'offline_wall_reduction_pct':wall_reduction,
        'ab_token_compare':ab_tokens,
        'projection_execution':{
            'pass':execution_pass,'rank0':s0,'rank1':s1,'same_native_library_sha':same_lib,
            'rank0_library':(d0.get('native_hip_identity') or {}).get('sha256'),
            'rank1_library':(d1.get('native_hip_identity') or {}).get('sha256'),
        },
        'smoke':{'pass':smoke_pass,'text':smoke.get('text'),'tokens':smoke.get('completion_token_count'),'finish_reason':smoke.get('finish_reason'),'ttft_s':smoke.get('client',{}).get('ttft_s'),'wall_s':smoke.get('client',{}).get('wall_s')},
        'coding':coding,
        'json':json_result,
        'reasoning_high':{'pass':reasoning_pass,'final_answer':ans,'text':reasoning.get('text'),'tokens':reasoning.get('completion_token_count'),'finish_reason':reasoning.get('finish_reason')},
    }
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(out,indent=2,ensure_ascii=False))
    raise SystemExit(0 if ok else 1)

if __name__=='__main__': main()
