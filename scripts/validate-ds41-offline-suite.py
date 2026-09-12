#!/usr/bin/env python3
"""Independent post-run validation for the preregistered DS41 offline suite."""
from __future__ import annotations
import argparse, ast, json, math
from pathlib import Path


def result_map(doc):
    return {r['label']: r for r in doc.get('results', [])}


def validate_code(text: str) -> dict:
    out={'status':'FAIL','parse':False,'function_found':False,'tests':[],'reason':None}
    try:
        tree=ast.parse(text)
        out['parse']=True
    except SyntaxError as e:
        out['reason']=f'SyntaxError: {e}'
        return out
    # Only a function definition (plus optional module docstring) is appropriate for this prompt.
    body=list(tree.body)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]
    if len(body)!=1 or not isinstance(body[0],ast.FunctionDef) or body[0].name!='first_missing_positive':
        out['reason']='expected exactly first_missing_positive function definition'
        return out
    out['function_found']=True
    forbidden=(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Lambda,ast.ClassDef,ast.AsyncFunctionDef,ast.With,ast.AsyncWith,ast.Try)
    if any(isinstance(n,forbidden) for n in ast.walk(body[0])):
        out['reason']='forbidden construct in submitted source'
        return out
    allowed_builtins={'len':len,'range':range,'enumerate':enumerate,'abs':abs,'min':min,'max':max}
    ns={'__builtins__':allowed_builtins}
    try:
        exec(compile(tree,'<ds41-answer>','exec'),ns,ns)
        fn=ns['first_missing_positive']
        cases=[([1,2,0],3),([3,4,-1,1],2),([7,8,9,11,12],1),([1],2),([],1),([2,1],3),([1,1,2,2],3),([-5,-1,4,2,1],3),([2,3,4,5,1],6)]
        ok=True
        for inp,expected in cases:
            arr=list(inp); got=fn(arr); good=(got==expected)
            out['tests'].append({'input':inp,'expected':expected,'got':got,'pass':good}); ok &= good
        out['status']='PASS' if ok else 'FAIL'
        if not ok: out['reason']='behavioral test failure'
    except BaseException as e:
        out['reason']=f'{type(e).__name__}: {e}'
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank0',required=True); ap.add_argument('--rank1',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    d0=json.load(open(a.rank0)); d1=json.load(open(a.rank1)); r0=result_map(d0); r1=result_map(d1)
    labels=sorted(set(r0)|set(r1)); token_identity={}
    for lab in labels:
        token_identity[lab]= lab in r0 and lab in r1 and r0[lab].get('token_ids')==r1[lab].get('token_ids')
    smoke=r0.get('smoke-arithmetic',{}); reasoning=r0.get('quality-reasoning',{}); js=r0.get('quality-json',{}); coding=r0.get('quality-coding',{})
    smoke_pass=bool(smoke) and smoke.get('finished') and smoke.get('text','').strip()=='323'
    reasoning_pass=bool(reasoning) and reasoning.get('finished') and reasoning.get('text','').strip()=='10'
    json_res={'status':'NOT_RUN'}
    if js:
        try:
            obj=json.loads(js.get('text',''))
            good=(isinstance(obj,dict) and set(obj)=={'total','valid_ids'} and obj['total']==17 and obj['valid_ids']==['a','c'])
            json_res={'status':'PASS' if good else 'FAIL','parsed':obj}
        except Exception as e: json_res={'status':'FAIL','error':f'{type(e).__name__}: {e}','text':js.get('text','')}
    coding_res={'status':'NOT_RUN'} if not coding else validate_code(coding.get('text',''))
    measured=[]
    for i in range(1,4):
        x=r0.get(f'speed-measured-{i}')
        if x:
            measured.append({'label':x['label'],'prompt_tokens':x.get('prompt_token_count'),'completion_tokens':x.get('completion_token_count'),'finish_reason':x.get('finish_reason'),'wall_s':x.get('client',{}).get('wall_s'),'ttft_s':x.get('client',{}).get('ttft_s'),'decode_tps':x.get('derived',{}).get('decode_tps_first_to_last')})
    tps=[x['decode_tps'] for x in measured if isinstance(x.get('decode_tps'),(int,float))]
    walls=[x['wall_s'] for x in measured if isinstance(x.get('wall_s'),(int,float))]
    ttfts=[x['ttft_s'] for x in measured if isinstance(x.get('ttft_s'),(int,float))]
    perf={'runs':measured,'decode_tps_mean':sum(tps)/len(tps) if tps else None,'decode_tps_min':min(tps) if tps else None,'decode_tps_max':max(tps) if tps else None,'wall_s_mean':sum(walls)/len(walls) if walls else None,'ttft_s_mean':sum(ttfts)/len(ttfts) if ttfts else None,'http_metrics':'N/A (offline SPMD)'}
    out={'status':'PASS' if smoke_pass and reasoning_pass and json_res.get('status')=='PASS' and coding_res.get('status')=='PASS' and all(token_identity.values()) else 'FAIL','rank0_status':d0.get('status'),'rank1_status':d1.get('status'),'token_identity':token_identity,'smoke':{'pass':smoke_pass,'text':smoke.get('text'),'completion_tokens':smoke.get('completion_token_count'),'finish_reason':smoke.get('finish_reason'),'wall_s':smoke.get('client',{}).get('wall_s'),'ttft_s':smoke.get('client',{}).get('ttft_s'),'decode_tps':smoke.get('derived',{}).get('decode_tps_first_to_last')},'reasoning':{'pass':reasoning_pass,'text':reasoning.get('text')},'json':json_res,'coding':coding_res,'performance':perf}
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(out,indent=2,ensure_ascii=False))
    raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__': main()
