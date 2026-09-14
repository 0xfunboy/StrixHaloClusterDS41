#!/usr/bin/env python3
"""Post-fix wrapper dispatch gate for the narrowly qualified T=3 mHC control."""
from __future__ import annotations
import importlib.util, json, os
from pathlib import Path
import torch

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
RANK=int(os.environ.get('DS41_DISCRIMINATOR_RANK',os.environ.get('RANK','0')))
OUT=Path(os.environ.get('DS41_MHC_T3_WRAPPER_OUT',str(ROOT/f'reports/DS41-Q2-001/attempt035/mhc-t3-wrapper-rank{RANK}.json')))
# Reuse the already-published pre-fix T3 qualification helpers.
spec=importlib.util.spec_from_file_location('t3gate',ROOT/'scripts/test-ds41-attempt035-mhc-t3.py')
t3=importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
disc=t3.disc
from runtime.ds41.mhc_projection_rms import reset_stats as pres, stats as pstats
from runtime.ds41.mhc_coeff_sinkhorn import reset_stats as cres, stats as cstats
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

NAMES=t3.NAMES

def call(residual,fn,scale,base,*,pre_mix=None,x=None,rowwise=True):
    old={k:os.environ.get(k) for k in ('DS41_MHC_PROJECTION_RMS','DS41_MHC_COEFF_SINKHORN','DS41_MHC_ROWWISE_BLOCK')}
    os.environ['DS41_MHC_PROJECTION_RMS']='1'; os.environ['DS41_MHC_COEFF_SINKHORN']='1'; os.environ['DS41_MHC_ROWWISE_BLOCK']='1' if rowwise else '0'
    pres(); cres()
    try:
        out=tuple(t.detach().clone() for t in mhc_pre_delayed_torch(
            residual,fn,scale,base,disc.EPS,disc.HC_EPS,disc.HC_EPS,disc.POST_MULT,disc.SINK_ITERS,
            pre_mix=pre_mix,x=x))
        st={'projection':pstats(),'coefficient':cstats()}
        return out,st
    finally:
        for k,v in old.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v

def metrics(a,b): return {n:disc.metric(x,y) for n,x,y in zip(NAMES,a,b,strict=True)}
def exact(g): return all(v.get('exact') for v in g.values())

def main():
    assert torch.cuda.is_available()
    broad={label:disc.load_boundary(disc.boundary_file(disc.RAW29,label,RANK)) for label in ('diagnostic-D1','diagnostic-B4')}
    narrow={label:disc.load_boundary(disc.boundary_file(disc.RAW31,label,RANK)) for label in ('diagnostic-D1','diagnostic-B4')}
    afn=disc.f32('blk.0.hc_attn_fn').reshape(24,disc.HC,disc.H).sum(1).contiguous(); asc=disc.f32('blk.0.hc_attn_scale'); abase=disc.f32('blk.0.hc_attn_base')
    ffn=disc.f32('blk.0.hc_ffn_fn'); fsc=disc.f32('blk.0.hc_ffn_scale'); fbase=disc.f32('blk.0.hc_ffn_base')
    rec=disc.reconstruct_ffn_inputs('diagnostic-B4',broad['diagnostic-B4'],narrow['diagnostic-B4'],afn,asc,abase)
    entry=broad['diagnostic-B4'][(0,'layer_entry')].to('cuda')[:3].contiguous(); rattn=entry[:,None,:].expand(-1,disc.HC,-1).contiguous()
    rffn=rec['residual_before_ffn_pre'][:3].contiguous(); pffn=rec['attention_pre'][:3].contiguous()
    cases={}
    for name,args,kwargs in (
        ('attention',(rattn,afn,asc,abase),{'x':entry}),
        ('ffn',(rffn,ffn,fsc,fbase),{'pre_mix':pffn}),
    ):
        ref=t3.concat_m1(*args,**kwargs)
        on,ston=call(*args,**kwargs,rowwise=True)
        off,stoff=call(*args,**kwargs,rowwise=False)
        cases[name]={
          'wrapper_on_vs_concat_m1':metrics(on,ref), 'wrapper_on_stats':ston,
          'wrapper_off_vs_concat_m1':metrics(off,ref), 'wrapper_off_stats':stoff,
        }
        if not exact(cases[name]['wrapper_on_vs_concat_m1']): raise RuntimeError(f'{name} T3 wrapper ON not exact')
        if ston['projection']['tilelang_calls']!=3 or ston['projection']['fallback_calls']!=0: raise RuntimeError(f'{name} T3 projection dispatch {ston}')
        if ston['coefficient']['fused_calls']!=1 or ston['coefficient']['fused_tokens']!=3 or ston['coefficient']['fallback_calls']!=0: raise RuntimeError(f'{name} T3 coeff dispatch {ston}')
        if stoff['projection']['tilelang_calls']!=0 or stoff['projection']['fallback_calls']!=1 or stoff['projection']['fallback_reasons'].get('tokens_not_1')!=1: raise RuntimeError(f'{name} OFF projection fallback changed {stoff}')
        if stoff['coefficient']['fused_calls']!=0 or stoff['coefficient']['fallback_calls']!=1 or stoff['coefficient']['fallback_reasons'].get('tokens_not_1')!=1: raise RuntimeError(f'{name} OFF coeff fallback changed {stoff}')
    report={'schema':'ds41-attempt035-mhc-t3-wrapper-v1','status':'PASS','rank':RANK,'tokens':3,'cases':cases,
            'guard_expected':'DS41_MHC_ROWWISE_BLOCK=1 and T in (2,3,4)','default_off_preserved':True}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
