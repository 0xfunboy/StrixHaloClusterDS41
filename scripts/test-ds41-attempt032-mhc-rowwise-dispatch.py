#!/usr/bin/env python3
"""Model-free dispatch gate for DS41 T=2/4 mHC fidelity alignment."""
from __future__ import annotations
import importlib.util, json, os
from pathlib import Path
import torch

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=Path(os.environ.get('DS41_MHC_ROWWISE_GATE_OUT',str(ROOT/'reports/DS41-Q2-001/attempt032/mhc-rowwise-gate-node01.json')))
RANK=int(os.environ.get('DS41_DISCRIMINATOR_RANK',os.environ.get('RANK','0')))

spec=importlib.util.spec_from_file_location('disc',ROOT/'scripts/test-ds41-attempt032-mhc-discriminator.py')
disc=importlib.util.module_from_spec(spec); spec.loader.exec_module(disc)
from runtime.ds41.mhc_projection_rms import reset_stats as pres, stats as pstats
from runtime.ds41.mhc_coeff_sinkhorn import reset_stats as cres, stats as cstats
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch


def clone_tuple(xs): return tuple(x.detach().clone() for x in xs)
def metrics_tuple(a,b):
    return {name:disc.metric(x,y) for name,x,y in zip(('post_mix','res_mix','layer_input','next_pre'),a,b,strict=True)}

def env_call(residual,fn,scale,base,*,pre_mix=None,x=None,rowwise=False,p=True,c=True):
    old={k:os.environ.get(k) for k in ('DS41_MHC_PROJECTION_RMS','DS41_MHC_COEFF_SINKHORN','DS41_MHC_ROWWISE_BLOCK')}
    os.environ['DS41_MHC_PROJECTION_RMS']='1' if p else '0'
    os.environ['DS41_MHC_COEFF_SINKHORN']='1' if c else '0'
    os.environ['DS41_MHC_ROWWISE_BLOCK']='1' if rowwise else '0'
    try:
        return clone_tuple(mhc_pre_delayed_torch(residual,fn,scale,base,disc.EPS,disc.HC_EPS,disc.HC_EPS,disc.POST_MULT,disc.SINK_ITERS,pre_mix=pre_mix,x=x))
    finally:
        for k,v in old.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v

def stats_call(*args,**kwargs):
    pres(); cres(); out=env_call(*args,**kwargs); return out,{'projection':pstats(),'coefficient':cstats()}

def load(label):
    return disc.load_boundary(disc.boundary_file(disc.RAW29,label,RANK))

def main():
    assert torch.cuda.is_available()
    labels=('diagnostic-D1','diagnostic-B2','diagnostic-B4')
    broad={x:load(x) for x in labels}
    afn=disc.f32('blk.0.hc_attn_fn').reshape(24,disc.HC,disc.H).sum(1).contiguous()
    asc=disc.f32('blk.0.hc_attn_scale'); abase=disc.f32('blk.0.hc_attn_base')
    ffn=disc.f32('blk.0.hc_ffn_fn'); fsc=disc.f32('blk.0.hc_ffn_scale'); fbase=disc.f32('blk.0.hc_ffn_base')
    # Reconstruct actual pre-FFN states with the pre-fix (rowwise OFF) attention semantics.
    narrow={label:disc.load_boundary(disc.boundary_file(disc.RAW31,label,RANK)) for label in labels}
    rec={label:disc.reconstruct_ffn_inputs(label,broad[label],narrow[label],afn,asc,abase) for label in labels}
    report={'schema':'ds41-attempt032-mhc-rowwise-dispatch-v1','status':'RUNNING','rank':RANK,'cases':{},'fail_closed':{}}

    # M1 target for initial attention and delayed FFN.
    e1=broad['diagnostic-D1'][(0,'layer_entry')].to('cuda').contiguous(); r1=e1[:,None,:].expand(-1,disc.HC,-1).contiguous()
    attn_ref,_=stats_call(r1,afn,asc,abase,x=e1,rowwise=False)
    ffn_ref,_=stats_call(rec['diagnostic-D1']['residual_before_ffn_pre'],ffn,fsc,fbase,pre_mix=rec['diagnostic-D1']['attention_pre'],rowwise=False)

    # Rowwise flag must not change M1.
    attn_m1_on,attn_m1_stats=stats_call(r1,afn,asc,abase,x=e1,rowwise=True)
    ffn_m1_on,ffn_m1_stats=stats_call(rec['diagnostic-D1']['residual_before_ffn_pre'],ffn,fsc,fbase,pre_mix=rec['diagnostic-D1']['attention_pre'],rowwise=True)
    report['m1_unchanged']={'attention':metrics_tuple(attn_m1_on,attn_ref),'ffn':metrics_tuple(ffn_m1_on,ffn_ref),'attention_stats':attn_m1_stats,'ffn_stats':ffn_m1_stats}

    for label in ('diagnostic-B2','diagnostic-B4'):
        entry=broad[label][(0,'layer_entry')].to('cuda').contiguous(); residual0=entry[:,None,:].expand(-1,disc.HC,-1).contiguous()
        attn_base,attn_base_stats=stats_call(residual0,afn,asc,abase,x=entry,rowwise=False)
        attn_fix,attn_fix_stats=stats_call(residual0,afn,asc,abase,x=entry,rowwise=True)
        ffn_base_out,ffn_base_stats=stats_call(rec[label]['residual_before_ffn_pre'],ffn,fsc,fbase,pre_mix=rec[label]['attention_pre'],rowwise=False)
        ffn_fix,ffn_fix_stats=stats_call(rec[label]['residual_before_ffn_pre'],ffn,fsc,fbase,pre_mix=rec[label]['attention_pre'],rowwise=True)
        report['cases'][label]={
            'tokens':int(entry.shape[0]),
            'attention_baseline_vs_m1_row0':metrics_tuple(tuple(x[0:1] for x in attn_base),attn_ref),
            'attention_rowwise_vs_m1_row0':metrics_tuple(tuple(x[0:1] for x in attn_fix),attn_ref),
            'ffn_baseline_vs_m1_row0':metrics_tuple(tuple(x[0:1] for x in ffn_base_out),ffn_ref),
            'ffn_rowwise_vs_m1_row0':metrics_tuple(tuple(x[0:1] for x in ffn_fix),ffn_ref),
            'attention_baseline_stats':attn_base_stats,'attention_rowwise_stats':attn_fix_stats,
            'ffn_baseline_stats':ffn_base_stats,'ffn_rowwise_stats':ffn_fix_stats,
        }
        # T2/T4 candidate dispatch: exactly T TileLang row calls, one fused coeff call, no fallback.
        T=int(entry.shape[0])
        for where,st in (('attention',attn_fix_stats),('ffn',ffn_fix_stats)):
            if st['projection']['tilelang_calls']!=T or st['projection']['fallback_calls']!=0:
                raise RuntimeError(f'{label} {where} projection dispatch mismatch: {st}')
            if st['coefficient']['fused_calls']!=1 or st['coefficient']['fused_tokens']!=T or st['coefficient']['fallback_calls']!=0:
                raise RuntimeError(f'{label} {where} coefficient dispatch mismatch: {st}')
        for pack in (report['cases'][label]['attention_rowwise_vs_m1_row0'],report['cases'][label]['ffn_rowwise_vs_m1_row0']):
            if not all(v.get('exact') for v in pack.values()):
                raise RuntimeError(f'{label} rowwise wrapper did not reproduce M1: {pack}')

    # Non-qualified T=7 remains exact fallback regardless of flag.
    b4=rec['diagnostic-B4']; res7=b4['residual_before_ffn_pre'][0:1].expand(7,-1,-1).contiguous(); pre7=b4['attention_pre'][0:1].expand(7,-1).contiguous()
    off7,st7off=stats_call(res7,ffn,fsc,fbase,pre_mix=pre7,rowwise=False)
    on7,st7on=stats_call(res7,ffn,fsc,fbase,pre_mix=pre7,rowwise=True)
    report['m7_fallback']={'metrics':metrics_tuple(on7,off7),'off_stats':st7off,'on_stats':st7on}
    if not all(v.get('exact') for v in report['m7_fallback']['metrics'].values()): raise RuntimeError('M7 fallback changed')

    # Fail closed if the rowwise fidelity flag is requested without both promoted components.
    b2=rec['diagnostic-B2'];
    for key,p,c in (('projection_off',False,True),('coeff_off',True,False)):
        try:
            env_call(b2['residual_before_ffn_pre'],ffn,fsc,fbase,pre_mix=b2['attention_pre'],rowwise=True,p=p,c=c)
        except RuntimeError as exc:
            report['fail_closed'][key]={'passed':True,'error':str(exc)}
        else:
            raise RuntimeError(f'fail-closed gate did not reject {key}')

    if not all(v.get('exact') for group in report['m1_unchanged'].values() if isinstance(group,dict) and 'post_mix' in group for v in group.values()):
        raise RuntimeError('M1 changed under rowwise flag')
    report['status']='PASS'; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
