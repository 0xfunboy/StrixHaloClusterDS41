#!/usr/bin/env python3
"""Model-free T=3 mHC qualification using real saved layer0 fixtures.

The runtime guard is intentionally NOT changed by this test.  It directly composes
three already-qualified M1 TileLang projection/RMS calls with one existing fused
coefficient/Sinkhorn [3,24] call and compares every row/output against three
independent promoted M1 wrapper calls on identical real inputs.
"""
from __future__ import annotations
import importlib.util, json, os, subprocess
from pathlib import Path
import torch

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
RANK=int(os.environ.get('DS41_DISCRIMINATOR_RANK',os.environ.get('RANK','0')))
OUT=Path(os.environ.get('DS41_MHC_T3_GATE_OUT',str(ROOT/f'reports/DS41-Q2-001/attempt035/mhc-t3-gate-rank{RANK}.json')))
spec=importlib.util.spec_from_file_location('disc',ROOT/'scripts/test-ds41-attempt032-mhc-discriminator.py')
disc=importlib.util.module_from_spec(spec); spec.loader.exec_module(disc)
from runtime.ds41.mhc_projection_rms import reset_stats as pres, stats as pstats
from runtime.ds41.mhc_coeff_sinkhorn import reset_stats as cres, stats as cstats, fused_coeff_sinkhorn
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

NAMES=('post_mix','res_mix','layer_input','next_pre')

def metric(a,b): return disc.metric(a,b)
def metrics_tuple(a,b): return {n:metric(x,y) for n,x,y in zip(NAMES,a,b,strict=True)}
def exact_group(g): return all(v.get('exact') for v in g.values())

def env_m1(residual,fn,scale,base,*,pre_mix=None,x=None):
    old={k:os.environ.get(k) for k in ('DS41_MHC_PROJECTION_RMS','DS41_MHC_COEFF_SINKHORN','DS41_MHC_ROWWISE_BLOCK')}
    os.environ['DS41_MHC_PROJECTION_RMS']='1'; os.environ['DS41_MHC_COEFF_SINKHORN']='1'; os.environ['DS41_MHC_ROWWISE_BLOCK']='0'
    try:
        return tuple(t.detach().clone() for t in mhc_pre_delayed_torch(
            residual,fn,scale,base,disc.EPS,disc.HC_EPS,disc.HC_EPS,disc.POST_MULT,disc.SINK_ITERS,
            pre_mix=pre_mix,x=x))
    finally:
        for k,v in old.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v

def concat_m1(residual,fn,scale,base,*,pre_mix=None,x=None):
    rows=[]
    for i in range(residual.shape[0]):
        rows.append(env_m1(
            residual[i:i+1].contiguous(),fn,scale,base,
            pre_mix=None if pre_mix is None else pre_mix[i:i+1].contiguous(),
            x=None if x is None else x[i:i+1].contiguous()))
    return tuple(torch.cat([r[j] for r in rows],dim=0).contiguous() for j in range(4))

def p1_c1_manual(residual,fn,scale,base,*,pre_mix=None,x=None):
    source=residual.flatten(1) if x is None else x
    pres(); cres()
    mixes=disc.rowwise_projection(source.contiguous(),fn)
    # One fused call across all three rows, never serialized.
    pre,post,comb=fused_coeff_sinkhorn(
        mixes.contiguous(),scale,base,disc.HC_EPS,disc.HC_EPS,disc.POST_MULT,disc.SINK_ITERS)
    layer_input=(residual[:,0] if pre_mix is None else
                 (pre_mix.unsqueeze(-1)*residual.float()).sum(dim=1).to(residual.dtype))
    out=(post.unsqueeze(-1),comb,layer_input,pre)
    stats={'projection':pstats(),'coefficient':cstats()}
    # Same mixes: one C1 T3 must equal three C1 T1 row programs exactly.
    concat=[]
    for i in range(3):
        concat.append(fused_coeff_sinkhorn(
            mixes[i:i+1].contiguous(),scale,base,disc.HC_EPS,disc.HC_EPS,disc.POST_MULT,disc.SINK_ITERS))
    c1cat=tuple(torch.cat([r[j] for r in concat],dim=0) for j in range(3))
    c1eq={n:metric(a,b) for n,a,b in zip(('pre','post','comb'),(pre,post,comb),c1cat,strict=True)}
    return out,stats,c1eq

def load(label,raw): return disc.load_boundary(disc.boundary_file(raw,label,RANK))

def main():
    assert torch.cuda.is_available()
    broad={label:load(label,disc.RAW29) for label in ('diagnostic-D1','diagnostic-B4')}
    narrow={label:load(label,disc.RAW31) for label in ('diagnostic-D1','diagnostic-B4')}
    afn=disc.f32('blk.0.hc_attn_fn').reshape(24,disc.HC,disc.H).sum(1).contiguous()
    asc=disc.f32('blk.0.hc_attn_scale'); abase=disc.f32('blk.0.hc_attn_base')
    ffn=disc.f32('blk.0.hc_ffn_fn'); fsc=disc.f32('blk.0.hc_ffn_scale'); fbase=disc.f32('blk.0.hc_ffn_base')
    rec=disc.reconstruct_ffn_inputs('diagnostic-B4',broad['diagnostic-B4'],narrow['diagnostic-B4'],afn,asc,abase)

    # Real K=5120 initial-attention fixture: first three rows of saved B4 packet.
    entry=broad['diagnostic-B4'][(0,'layer_entry')].to('cuda')[:3].contiguous()
    residual_attn=entry[:,None,:].expand(-1,disc.HC,-1).contiguous()
    attn_ref=concat_m1(residual_attn,afn,asc,abase,x=entry)
    attn_cand,attn_stats,attn_c1=p1_c1_manual(residual_attn,afn,asc,abase,x=entry)

    # Real K=20480 delayed-FFN fixture on the corresponding first three rows.
    residual_ffn=rec['residual_before_ffn_pre'][:3].contiguous()
    pre_ffn=rec['attention_pre'][:3].contiguous()
    ffn_ref=concat_m1(residual_ffn,ffn,fsc,fbase,pre_mix=pre_ffn)
    ffn_cand,ffn_stats,ffn_c1=p1_c1_manual(residual_ffn,ffn,fsc,fbase,pre_mix=pre_ffn)

    report={
      'schema':'ds41-attempt035-mhc-t3-gate-v1','status':'RUNNING','rank':RANK,
      'analysis_source':os.environ.get('DS41_ANALYSIS_SOURCE','unknown'),
      'runtime_marker':(ROOT/'.source-commit').read_text().strip() if (ROOT/'.source-commit').exists() else None,
      'model_initialized':False,'generation_requests':0,'tokens':3,
      'attention':{'source_shape':list(entry.shape),'fn_shape':list(afn.shape),'candidate_vs_concat_m1_all_rows':metrics_tuple(attn_cand,attn_ref),'candidate_stats':attn_stats,'C1_T3_vs_concat_T1':attn_c1},
      'ffn':{'residual_shape':list(residual_ffn.shape),'source_shape':list(residual_ffn.flatten(1).shape),'fn_shape':list(ffn.shape),'pre_mix_shape':list(pre_ffn.shape),'candidate_vs_concat_m1_all_rows':metrics_tuple(ffn_cand,ffn_ref),'candidate_stats':ffn_stats,'C1_T3_vs_concat_T1':ffn_c1},
      'contract':{'projection':'three existing qualified M1 TileLang calls','coefficient':'one existing fused [3,24] call','collective_per_row':False,'runtime_guard_modified_for_test':False},
    }
    for where in ('attention','ffn'):
        if not exact_group(report[where]['candidate_vs_concat_m1_all_rows']):
            raise RuntimeError(f'{where} T3 P1+C1 != concatenated M1: {report[where]["candidate_vs_concat_m1_all_rows"]}')
        if not exact_group(report[where]['C1_T3_vs_concat_T1']):
            raise RuntimeError(f'{where} C1 T3 != concat T1: {report[where]["C1_T3_vs_concat_T1"]}')
        st=report[where]['candidate_stats']
        if st['projection']['tilelang_calls']!=3 or st['projection']['tilelang_tokens']!=3 or st['projection']['fallback_calls']!=0:
            raise RuntimeError(f'{where} projection stats mismatch: {st}')
        if st['coefficient']['fused_calls']!=1 or st['coefficient']['fused_tokens']!=3 or st['coefficient']['fallback_calls']!=0:
            raise RuntimeError(f'{where} coeff stats mismatch: {st}')
    report['status']='PASS'; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
