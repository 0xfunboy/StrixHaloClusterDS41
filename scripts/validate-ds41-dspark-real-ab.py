#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, json, statistics, hashlib
from pathlib import Path

def rm(d): return {r['label']:r for r in d['results']}
def first_diff(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
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

def perf(rows, arm):
    runs=[]
    for i in (1,2,3):
        r=rows[f'dspark-real-{arm}-speed-{i}']
        runs.append({'trial':i,'tps':r['derived']['decode_tps_first_to_last'],'wall_s':r['client']['wall_s'],'ttft_s':r['client']['ttft_s'],'tokens':r['completion_token_count'],'finish_reason':r['finish_reason']})
    t=[x['tps'] for x in runs]; w=[x['wall_s'] for x in runs]; f=[x['ttft_s'] for x in runs]
    return {'runs':runs,'mean_tps':statistics.mean(t),'median_tps':statistics.median(t),'sample_sd_tps':statistics.stdev(t),'cv_pct':statistics.stdev(t)/statistics.mean(t)*100,'mean_wall_s':statistics.mean(w),'median_wall_s':statistics.median(w),'mean_ttft_s':statistics.mean(f)}

def quality(rows, arm):
    ar=rows[f'dspark-real-{arm}-arithmetic']; js=rows[f'dspark-real-{arm}-json']; cd=rows[f'dspark-real-{arm}-coding']; rh=rows[f'dspark-real-{arm}-reasoning-high']
    try: obj=json.loads(js['text']); jp=obj=={'total':17,'valid_ids':['a','c']}
    except Exception as e: obj={'error':str(e)}; jp=False
    t=rh['text']; ans=t.rsplit('</think>',1)[-1].strip() if '</think>' in t else t.strip()
    return {'arithmetic_pass':ar['text'].strip()=='323','json_pass':jp,'json':obj,'coding':code_validate(cd['text']),'reasoning_high_pass':ans=='10','reasoning_high_final':ans}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a0',required=True); ap.add_argument('--a1',required=True); ap.add_argument('--b0',required=True); ap.add_argument('--b1',required=True); ap.add_argument('--a-spec',required=True); ap.add_argument('--b-spec',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    A0=json.load(open(a.a0)); A1=json.load(open(a.a1)); B0=json.load(open(a.b0)); B1=json.load(open(a.b1)); ar0,ar1,br0,br1=map(rm,(A0,A1,B0,B1))
    suffixes=['warmup-excluded','functional-excluded','speed-1','speed-2','speed-3','arithmetic','coding','json','reasoning-high']
    rank={}
    cross={}
    for s in suffixes:
        al=f'dspark-real-m1-{s}'; bl=f'dspark-real-dspark-{s}'
        rank[s]={'a_rank_equal':ar0[al]['token_ids']==ar1[al]['token_ids'],'b_rank_equal':br0[bl]['token_ids']==br1[bl]['token_ids']}
        cross[s]={'tokens_equal':ar0[al]['token_ids']==br0[bl]['token_ids'],'text_equal':ar0[al]['text']==br0[bl]['text'],'finish_reason_equal':ar0[al].get('finish_reason')==br0[bl].get('finish_reason'),'stop_reason_equal':ar0[al].get('stop_reason')==br0[bl].get('stop_reason'),'first_diff_index':first_diff(ar0[al]['token_ids'],br0[bl]['token_ids'])}
    aspec=json.load(open(a.a_spec)); bspec=json.load(open(a.b_spec))
    prompts_equal=aspec['prompts']==bspec['prompts']
    A=perf(ar0,'m1'); B=perf(br0,'dspark')
    speedB=[br0[f'dspark-real-dspark-speed-{i}'] for i in (1,2,3)]
    accepted=sum(x['spec_decode_metrics']['num_accepted_draft_tokens'] for x in speedB); drafted=sum(x['spec_decode_metrics']['num_draft_tokens'] for x in speedB)
    calls=sum(x['dspark_draft_timing']['proposal_calls'] for x in speedB); gpu=sum(x['dspark_draft_timing']['gpu_stream_ms_total'] for x in speedB); cpu=sum(x['dspark_draft_timing']['cpu_enqueue_ms_total'] for x in speedB)
    functional=br0['dspark-real-dspark-functional-excluded']; fm=functional['spec_decode_metrics']; seq=fm['per_step_accepted']; reject_then_resume=any(x==0 and 1 in seq[i+1:] for i,x in enumerate(seq))
    no_a_spec=all(ar0[f'dspark-real-m1-{s}'].get('spec_decode_metrics') is None and ar0[f'dspark-real-m1-{s}'].get('dspark_draft_timing') is None for s in suffixes)
    qa=quality(ar0,'m1'); qb=quality(br0,'dspark')
    all_equal=all(x['tokens_equal'] and x['finish_reason_equal'] and x['stop_reason_equal'] for x in cross.values())
    ranks_ok=all(x['a_rank_equal'] and x['b_rank_equal'] for x in rank.values())
    speed_protocol=all(x['tokens']==128 and x['finish_reason']=='length' for x in A['runs']+B['runs'])
    gain=(B['mean_tps']/A['mean_tps']-1)*100
    wall_reduction=(1-B['mean_wall_s']/A['mean_wall_s'])*100
    wall_throughput=(A['mean_wall_s']/B['mean_wall_s']-1)*100
    out={'schema':'ds41-real-dspark-ab-v1','status':'PASS' if prompts_equal and ranks_ok and all_equal and speed_protocol and no_a_spec and qa['coding']['status']=='PASS' and qb['coding']['status']=='PASS' and qa['arithmetic_pass'] and qb['arithmetic_pass'] and qa['json_pass'] and qb['json_pass'] and qa['reasoning_high_pass'] and qb['reasoning_high_pass'] and reject_then_resume else 'FAIL',
         'execution_order':['B039 DSpark K1','A040 M1'],'separate_loads':True,'same_load':False,'alternated':False,'prompt_panel_equal':prompts_equal,'rank_coherence':rank,'a_b_output_equality':cross,'speed_protocol_pass':speed_protocol,'A040':A,'B039':B,'gain_decode_tps_pct':gain,'wall_reduction_pct':wall_reduction,'wall_throughput_gain_pct':wall_throughput,'ttft_delta_ms':(B['mean_ttft_s']-A['mean_ttft_s'])*1000,'quality_A':qa,'quality_B':qb,
         'B_speed_acceptance':{'accepted':accepted,'drafted':drafted,'rejected':drafted-accepted,'rate':accepted/drafted,'same_prompt_replicates':True},
         'B_functional_recovery':{'accepted':fm['num_accepted_draft_tokens'],'drafted':fm['num_draft_tokens'],'rejected':fm['num_draft_tokens']-fm['num_accepted_draft_tokens'],'completion_tokens':functional['completion_token_count'],'reject_then_later_accept_observed':reject_then_resume,'per_step_accepted':seq},
         'B_proposer_timing':{'proposal_calls':calls,'gpu_stream_ms_total':gpu,'gpu_stream_ms_per_call':gpu/calls,'cpu_enqueue_ms_total_nonadditive':cpu,'cpu_enqueue_ms_per_call_nonadditive':cpu/calls,'gpu_span_pct_of_client_wall_nonadditive':100*(gpu/1000)/sum(x['client']['wall_s'] for x in speedB),'note':'GPU event span is around real DSpark propose on its CUDA stream. No synchronize inside timed request. CPU enqueue and GPU spans can overlap and are not added to client wall.'},
         'A_has_no_speculation':no_a_spec,
         'decode_tps_definition':'(final emitted completion_tokens - 1)/(last_token_ts - first_token_ts); draft proposals/rejections are not counted as output tokens',
         'numeric_logit_gate':'NOT_RECAPTURED in 039/040; target verifier math/gates are unchanged from attempt036 exact qualification. New A/B gate is exact emitted-token equality across every frozen greedy request.',
         'a_spec_sha256':hashlib.sha256(Path(a.a_spec).read_bytes()).hexdigest(),'b_spec_sha256':hashlib.sha256(Path(a.b_spec).read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(out,indent=2,ensure_ascii=False)); raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__': main()
