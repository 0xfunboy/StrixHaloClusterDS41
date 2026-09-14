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
        x=r[f'dspark-real-dspark-speed-{i}']
        runs.append({'trial':i,'tps':x['derived']['decode_tps_first_to_last'],'wall_s':x['client']['wall_s'],'ttft_s':x['client']['ttft_s'],'tokens':x['completion_token_count'],'finish_reason':x['finish_reason']})
    t=[x['tps'] for x in runs]
    return {'runs':runs,'mean_tps':statistics.mean(t),'median_tps':statistics.median(t),'sample_sd_tps':statistics.stdev(t),'cv_pct':statistics.stdev(t)/statistics.mean(t)*100,'mean_wall_s':statistics.mean(x['wall_s'] for x in runs),'mean_ttft_s':statistics.mean(x['ttft_s'] for x in runs)}

def quality(r):
    ar=r['dspark-real-dspark-arithmetic']; js=r['dspark-real-dspark-json']; cd=r['dspark-real-dspark-coding']; rh=r['dspark-real-dspark-reasoning-high']; tx=r['dspark-real-dspark-text']
    try: obj=json.loads(js['text']); jp=obj=={'total':17,'valid_ids':['a','c']}
    except Exception as e: obj={'error':str(e)}; jp=False
    ans=rh['text'].rsplit('</think>',1)[-1].strip() if '</think>' in rh['text'] else rh['text'].strip()
    return {'arithmetic_pass':ar['text'].strip()=='323','json_pass':jp,'json':obj,'coding':code_validate(cd['text']),'reasoning_high_pass':ans=='10','reasoning_high_final':ans,'text_finished':bool(tx.get('finished')),'text_tokens':tx['completion_token_count']}

def acceptance(r,k):
    speed=[r[f'dspark-real-dspark-speed-{i}'] for i in (1,2,3)]
    drafted=[]; accepted=[]
    for x in speed:
        m=x['spec_decode_metrics']; drafted += list(m['per_step_drafted']); accepted += list(m['per_step_accepted'])
    assert len(drafted)==len(accepted)
    total_d=sum(drafted); total_a=sum(accepted)
    full=[(d,a) for d,a in zip(drafted,accepted) if d==k]
    tails=[(d,a) for d,a in zip(drafted,accepted) if d<k]
    full_hist={str(j):sum(1 for _,a in full if a==j) for j in range(k+1)}
    at_least_first=sum(1 for _,a in full if a>=1)
    second_cond=(sum(1 for _,a in full if a>=2)/at_least_first) if k>=2 and at_least_first else None
    full_accept=sum(1 for _,a in full if a==k)/len(full) if full else None
    verify_calls=len(drafted)
    completion=sum(x['completion_token_count'] for x in speed)
    return {'drafted':total_d,'accepted':total_a,'rejected':total_d-total_a,'overall_acceptance_rate':total_a/total_d if total_d else None,'verify_calls':verify_calls,'final_emitted_tokens':completion,'final_emitted_tokens_per_verify_call':completion/verify_calls if verify_calls else None,'full_width_steps':len(full),'full_width_accept_histogram':full_hist,'first_proposal_acceptance_rate_full_width':at_least_first/len(full) if full else None,'second_acceptance_conditional_on_first_full_width':second_cond,'full_block_acceptance_rate':full_accept,'tail_steps':len(tails),'tail_drafted_accepted':tails}

def proposer(r):
    speed=[r[f'dspark-real-dspark-speed-{i}'] for i in (1,2,3)]
    calls=sum(x['dspark_draft_timing']['proposal_calls'] for x in speed)
    gpu=sum(x['dspark_draft_timing']['gpu_stream_ms_total'] for x in speed)
    cpu=sum(x['dspark_draft_timing']['cpu_enqueue_ms_total'] for x in speed)
    return {'proposal_calls':calls,'gpu_stream_ms_total':gpu,'gpu_stream_ms_per_call':gpu/calls,'cpu_enqueue_ms_total_nonadditive':cpu,'cpu_enqueue_ms_per_call_nonadditive':cpu/calls}

def main():
    ap=argparse.ArgumentParser()
    for x in ('k1_0','k1_1','k2_0','k2_1','k1_spec','k2_spec','output'): ap.add_argument('--'+x.replace('_','-'),dest=x,required=True)
    a=ap.parse_args()
    K10,K11,K20,K21=[json.load(open(x)) for x in (a.k1_0,a.k1_1,a.k2_0,a.k2_1)]
    r10,r11,r20,r21=map(rows,(K10,K11,K20,K21)); s1=json.load(open(a.k1_spec)); s2=json.load(open(a.k2_spec))
    suffix=['warmup-excluded','functional-excluded','speed-1','speed-2','speed-3','arithmetic','coding','text','json','reasoning-high']
    rank={}; cross={}
    for s in suffix:
        lab=f'dspark-real-dspark-{s}'
        rank[s]={'k1':r10[lab]['token_ids']==r11[lab]['token_ids'] and r10[lab]['text']==r11[lab]['text'],'k2':r20[lab]['token_ids']==r21[lab]['token_ids'] and r20[lab]['text']==r21[lab]['text']}
        cross[s]={'tokens_equal':r10[lab]['token_ids']==r20[lab]['token_ids'],'text_equal':r10[lab]['text']==r20[lab]['text'],'finish_reason_equal':r10[lab].get('finish_reason')==r20[lab].get('finish_reason'),'stop_reason_equal':r10[lab].get('stop_reason')==r20[lab].get('stop_reason'),'first_diff_index':first_diff(r10[lab]['token_ids'],r20[lab]['token_ids'])}
    prompts_equal=s1['prompts']==s2['prompts']
    cfg_ok=(K10.get('configured_num_speculative_tokens')==1 and K11.get('configured_num_speculative_tokens')==1 and K20.get('configured_num_speculative_tokens')==2 and K21.get('configured_num_speculative_tokens')==2)
    P1,P2=perf(r10),perf(r20); Q1,Q2=quality(r10),quality(r20)
    gain=(P2['mean_tps']/P1['mean_tps']-1)*100
    median_win=P2['median_tps']>P1['median_tps']
    speed_wins=sum(b['tps']>a0['tps'] for a0,b in zip(P1['runs'],P2['runs']))
    control_labels=['coding','text','reasoning-high']
    c1=sum(r10[f'dspark-real-dspark-{x}']['client']['wall_s'] for x in control_labels)
    c2=sum(r20[f'dspark-real-dspark-{x}']['client']['wall_s'] for x in control_labels)
    control_reg=(c2/c1-1)*100
    A1,A2=acceptance(r10,1),acceptance(r20,2); D1,D2=proposer(r10),proposer(r20)
    ranks_ok=all(v['k1'] and v['k2'] for v in rank.values())
    equality=all(v['tokens_equal'] and v['text_equal'] and v['finish_reason_equal'] and v['stop_reason_equal'] for v in cross.values())
    quality_ok=all([Q1['arithmetic_pass'],Q2['arithmetic_pass'],Q1['json_pass'],Q2['json_pass'],Q1['reasoning_high_pass'],Q2['reasoning_high_pass'],Q1['coding']['status']=='PASS',Q2['coding']['status']=='PASS',Q1['text_finished'],Q2['text_finished']])
    perf_pass=(gain>=5.0 and median_win and speed_wins>=2 and control_reg<=10.0)
    dispatch_ok=(K10.get('runtime_switches',{}).get('DS41_NATIVE_HIP_MOE_ROWWISE')=='1' and K20.get('runtime_switches',{}).get('DS41_NATIVE_HIP_MOE_ROWWISE')=='1' and K10.get('runtime_switches',{}).get('DS41_MHC_ROWWISE_BLOCK')=='1' and K20.get('runtime_switches',{}).get('DS41_MHC_ROWWISE_BLOCK')=='1')
    correctness_pass=prompts_equal and cfg_ok and ranks_ok and equality and quality_ok and dispatch_ok
    status='PASS' if correctness_pass and perf_pass else ('CORRECTNESS_PASS_PERF_FAIL' if correctness_pass else 'FAIL')
    out={'schema':'ds41-dspark-k2-vs-k1-v1','status':status,'decision':'QUALIFY_K2_EXPERIMENTAL' if status=='PASS' else 'KEEP_K1','execution_order':['B041 K2','A042 K1'],'separate_loads':True,'prompt_panel_equal':prompts_equal,'config_k_ok':cfg_ok,'rank_coherence':rank,'k1_k2_output_equality':cross,'K1':P1,'K2':P2,'gain_decode_tps_pct':gain,'median_win':median_win,'same_index_speed_wins':speed_wins,'distinct_code_text_reasoning_wall':{'k1_total_s':c1,'k2_total_s':c2,'k2_regression_pct':control_reg,'gate_max_pct':10.0},'performance_gate':{'mean_gain_min_pct':5.0,'median_k2_gt_k1':True,'speed_wins_min':2,'control_wall_regression_max_pct':10.0,'pass':perf_pass},'quality_K1':Q1,'quality_K2':Q2,'acceptance_K1':A1,'acceptance_K2':A2,'proposer_K1':D1,'proposer_K2':D2,'target_dispatch_switches_pass':dispatch_ok,'target_dispatch_stats_K1':K10.get('target_dispatch_stats'),'target_dispatch_stats_K2':K20.get('target_dispatch_stats'),'memory_K1':K10.get('load_resources'),'memory_K2':K20.get('load_resources'),'decode_tps_definition':'(final emitted completion_tokens - 1)/(last_token_ts - first_token_ts); draft proposals/rejections are excluded','gpu_proposer_note':'CUDA-event stream span around real propose; CPU enqueue may overlap and is not added to wall','k1_spec_sha256':hashlib.sha256(Path(a.k1_spec).read_bytes()).hexdigest(),'k2_spec_sha256':hashlib.sha256(Path(a.k2_spec).read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(out,indent=2,ensure_ascii=False)); raise SystemExit(0 if status=='PASS' else 1)
if __name__=='__main__': main()
