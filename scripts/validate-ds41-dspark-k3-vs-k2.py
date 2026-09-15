#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, statistics
from pathlib import Path

def rows(d): return {r['label']:r for r in d['results']}
def first_diff(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return None if len(a)==len(b) else min(len(a),len(b))
def code_validate(text):
    out={'status':'FAIL','tests':[]}
    try: tree=ast.parse(text)
    except Exception as e: out['error']=f'{type(e).__name__}: {e}'; return out
    body=list(tree.body)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str): body=body[1:]
    if len(body)!=1 or not isinstance(body[0],ast.FunctionDef) or body[0].name!='first_missing_positive': out['error']='unexpected module shape'; return out
    forbidden=(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Lambda,ast.ClassDef,ast.AsyncFunctionDef,ast.With,ast.AsyncWith,ast.Try)
    if any(isinstance(n,forbidden) for n in ast.walk(body[0])): out['error']='forbidden construct'; return out
    ns={'__builtins__':{'len':len,'range':range,'enumerate':enumerate,'abs':abs,'min':min,'max':max}}
    try: exec(compile(tree,'<saved-answer>','exec'),ns,ns)
    except BaseException as e: out['error']=f'{type(e).__name__}: {e}'; return out
    cases=[([1,2,0],3),([3,4,-1,1],2),([7,8,9,11,12],1),([1],2),([],1),([2,1],3),([1,1,2,2],3),([-5,-1,4,2,1],3),([2,3,4,5,1],6)]
    for inp,exp in cases:
        try: got=ns['first_missing_positive'](list(inp)); good=got==exp
        except BaseException as e: got=f'{type(e).__name__}: {e}'; good=False
        out['tests'].append({'input':inp,'expected':exp,'got':got,'pass':good})
    out['status']='PASS' if all(x['pass'] for x in out['tests']) else 'FAIL'; return out
def perf(r):
    runs=[]
    for i in (1,2,3):
        x=r[f'dspark-real-dspark-speed-{i}']; runs.append({'trial':i,'tps':x['derived']['decode_tps_first_to_last'],'wall_s':x['client']['wall_s'],'ttft_s':x['client']['ttft_s'],'tokens':x['completion_token_count'],'finish_reason':x['finish_reason']})
    t=[x['tps'] for x in runs]
    return {'runs':runs,'mean_tps':statistics.mean(t),'median_tps':statistics.median(t),'sample_sd_tps':statistics.stdev(t),'cv_pct':statistics.stdev(t)/statistics.mean(t)*100,'mean_wall_s':statistics.mean(x['wall_s'] for x in runs),'mean_ttft_s':statistics.mean(x['ttft_s'] for x in runs)}
def quality(r):
    ar=r['dspark-real-dspark-arithmetic']; js=r['dspark-real-dspark-json']; cd=r['dspark-real-dspark-coding']; rh=r['dspark-real-dspark-reasoning-high']; tx=r['dspark-real-dspark-text']
    try: obj=json.loads(js['text']); jp=obj=={'total':17,'valid_ids':['a','c']}
    except Exception as e: obj={'error':str(e)}; jp=False
    ans=rh['text'].rsplit('</think>',1)[-1].strip() if '</think>' in rh['text'] else rh['text'].strip()
    return {'arithmetic_pass':ar['text'].strip()=='323','json_pass':jp,'json':obj,'coding':code_validate(cd['text']),'reasoning_high_pass':ans=='10','reasoning_high_final':ans,'text_finished':bool(tx.get('finished')),'text_tokens':tx['completion_token_count']}
def acceptance(r,k):
    speed=[r[f'dspark-real-dspark-speed-{i}'] for i in (1,2,3)]; drafted=[]; accepted=[]
    for x in speed:
        m=x['spec_decode_metrics']; drafted += list(m['per_step_drafted']); accepted += list(m['per_step_accepted'])
    full=[(d,a) for d,a in zip(drafted,accepted) if d==k]; tails=[(d,a) for d,a in zip(drafted,accepted) if d<k]
    hist={str(j):sum(1 for _,a in full if a==j) for j in range(k+1)}
    ge1=sum(a>=1 for _,a in full); ge2=sum(a>=2 for _,a in full); ge3=sum(a>=3 for _,a in full)
    calls=len(drafted); completion=sum(x['completion_token_count'] for x in speed); bootstrap=len(speed); verified=completion-bootstrap
    return {'drafted':sum(drafted),'accepted':sum(accepted),'rejected':sum(drafted)-sum(accepted),'overall_acceptance_rate':sum(accepted)/sum(drafted) if sum(drafted) else None,'verify_calls':calls,'final_emitted_tokens_total':completion,'bootstrap_tokens':bootstrap,'verified_emitted_tokens':verified,'verified_emitted_tokens_per_verify_call':verified/calls if calls else None,'full_width_steps':len(full),'full_width_accept_histogram':hist,'first_proposal_acceptance_rate_full_width':ge1/len(full) if full else None,'second_acceptance_conditional_on_first_full_width':ge2/ge1 if ge1 else None,'third_acceptance_conditional_on_first_two_full_width':ge3/ge2 if k>=3 and ge2 else None,'full_block_acceptance_rate':sum(a==k for _,a in full)/len(full) if full else None,'tail_steps':len(tails),'tail_drafted_accepted':tails}
def proposer(r):
    speed=[r[f'dspark-real-dspark-speed-{i}'] for i in (1,2,3)]; calls=sum(x['dspark_draft_timing']['proposal_calls'] for x in speed); gpu=sum(x['dspark_draft_timing']['gpu_stream_ms_total'] for x in speed); cpu=sum(x['dspark_draft_timing']['cpu_enqueue_ms_total'] for x in speed)
    return {'proposal_calls':calls,'gpu_stream_ms_total':gpu,'gpu_stream_ms_per_call':gpu/calls,'cpu_enqueue_ms_total_nonadditive':cpu,'cpu_enqueue_ms_per_call_nonadditive':cpu/calls}
def main():
    ap=argparse.ArgumentParser()
    for x in ('k2_0','k2_1','k3_0','k3_1','k2_spec','k3_spec','output'): ap.add_argument('--'+x.replace('_','-'),dest=x,required=True)
    a=ap.parse_args(); K20,K21,K30,K31=[json.load(open(x)) for x in (a.k2_0,a.k2_1,a.k3_0,a.k3_1)]; r20,r21,r30,r31=map(rows,(K20,K21,K30,K31)); s2=json.load(open(a.k2_spec)); s3=json.load(open(a.k3_spec))
    suffix=['warmup-excluded','functional-excluded','speed-1','speed-2','speed-3','arithmetic','coding','text','json','reasoning-high']; rank={}; cross={}
    for s in suffix:
        lab=f'dspark-real-dspark-{s}'; rank[s]={'k2':r20[lab]['token_ids']==r21[lab]['token_ids'] and r20[lab]['text']==r21[lab]['text'],'k3':r30[lab]['token_ids']==r31[lab]['token_ids'] and r30[lab]['text']==r31[lab]['text']}; cross[s]={'tokens_equal':r20[lab]['token_ids']==r30[lab]['token_ids'],'text_equal':r20[lab]['text']==r30[lab]['text'],'finish_reason_equal':r20[lab].get('finish_reason')==r30[lab].get('finish_reason'),'stop_reason_equal':r20[lab].get('stop_reason')==r30[lab].get('stop_reason'),'first_diff_index':first_diff(r20[lab]['token_ids'],r30[lab]['token_ids'])}
    prompts_equal=s2['prompts']==s3['prompts']; cfg_ok=all(x.get('configured_num_speculative_tokens')==k for x,k in ((K20,2),(K21,2),(K30,3),(K31,3)))
    P2,P3=perf(r20),perf(r30); Q2,Q3=quality(r20),quality(r30); gain=(P3['mean_tps']/P2['mean_tps']-1)*100; median=P3['median_tps']>P2['median_tps']; wins=sum(b['tps']>aa['tps'] for aa,b in zip(P2['runs'],P3['runs']))
    labels=['coding','text','reasoning-high']; c2=sum(r20[f'dspark-real-dspark-{x}']['client']['wall_s'] for x in labels); c3=sum(r30[f'dspark-real-dspark-{x}']['client']['wall_s'] for x in labels); reg=(c3/c2-1)*100
    A2,A3=acceptance(r20,2),acceptance(r30,3); D2,D3=proposer(r20),proposer(r30); ranks_ok=all(v['k2'] and v['k3'] for v in rank.values()); equality=all(all(v[x] for x in ('tokens_equal','text_equal','finish_reason_equal','stop_reason_equal')) for v in cross.values()); qok=all([Q2['arithmetic_pass'],Q3['arithmetic_pass'],Q2['json_pass'],Q3['json_pass'],Q2['reasoning_high_pass'],Q3['reasoning_high_pass'],Q2['coding']['status']=='PASS',Q3['coding']['status']=='PASS',Q2['text_finished'],Q3['text_finished']]); dispatch=all(x.get('runtime_switches',{}).get('DS41_NATIVE_HIP_MOE_ROWWISE')=='1' and x.get('runtime_switches',{}).get('DS41_MHC_ROWWISE_BLOCK')=='1' for x in (K20,K30)); perf_pass=gain>=5 and median and wins==3 and reg<=10; correct=prompts_equal and cfg_ok and ranks_ok and equality and qok and dispatch; status='PASS' if correct and perf_pass else ('CORRECTNESS_PASS_PERF_FAIL' if correct else 'FAIL')
    out={'schema':'ds41-dspark-k3-vs-k2-v1','status':status,'decision':'QUALIFY_K3_EXPERIMENTAL' if status=='PASS' else ('NO_PROMOTION_KEEP_K2' if correct else 'CORRECTNESS_FAIL_KEEP_K2'),'execution_order':['B K3','A K2'],'separate_loads':True,'prompt_panel_equal':prompts_equal,'config_k_ok':cfg_ok,'rank_coherence':rank,'k2_k3_output_equality':cross,'K2':P2,'K3':P3,'gain_decode_tps_pct':gain,'median_win':median,'same_index_speed_wins':wins,'distinct_code_text_reasoning_wall':{'k2_total_s':c2,'k3_total_s':c3,'k3_regression_pct':reg,'gate_max_pct':10.0},'performance_gate':{'mean_gain_min_pct':5.0,'median_k3_gt_k2':True,'speed_wins_required':3,'control_wall_regression_max_pct':10.0,'pass':perf_pass},'quality_K2':Q2,'quality_K3':Q3,'acceptance_K2':A2,'acceptance_K3':A3,'proposer_K2':D2,'proposer_K3':D3,'target_dispatch_switches_pass':dispatch,'target_dispatch_stats_K2':K20.get('target_dispatch_stats'),'target_dispatch_stats_K3':K30.get('target_dispatch_stats'),'memory_K2':K20.get('load_resources'),'memory_K3':K30.get('load_resources'),'decode_tps_definition':'(final emitted completion_tokens - 1)/(last_token_ts - first_token_ts); drafts/rejections excluded','bootstrap_note':'verified_emitted_tokens subtracts exactly one bootstrap token per independent speed request before per-verify-call aggregation','gpu_proposer_note':'CUDA-event stream span; CPU enqueue may overlap and is not added to wall','k2_spec_sha256':hashlib.sha256(Path(a.k2_spec).read_bytes()).hexdigest(),'k3_spec_sha256':hashlib.sha256(Path(a.k3_spec).read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(out,indent=2,ensure_ascii=False)); raise SystemExit(0 if status=='PASS' else 1)
if __name__=='__main__': main()
