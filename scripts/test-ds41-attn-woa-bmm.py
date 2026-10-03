#!/usr/bin/env python3
"""DS41 model-free WO_A candidate gate on real layer0 attention output.

Compares the pinned ROCm `torch.einsum(tgd,grd->tgr)` projection against a
semantically identical grouped `torch.bmm` after the same fused inverse RoPE.
No cache, attention, WO_B or TP behavior is changed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=Path(os.environ.get('DS41_WOA_BMM_OUT',str(ROOT/'reports/DS41-Q2-001/perf/attention-woa-bmm-node01.json')))
REPEATS=int(os.environ.get('DS41_WOA_BMM_REPEATS','300'))
REL_MAX=5e-4
ABS_MAX=1.25e-1


def load_l0():
    p=ROOT/'scripts/test-ds41-layer0-complete-densefix.py'
    s=importlib.util.spec_from_file_location('ds41_l0',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def metric(a,b):
    af,bf=a.float(),b.float(); d=af-bf; rn=float(bf.norm())
    return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'rel_l2':float(d.norm())/max(rn,1e-30),'finite':bool(torch.isfinite(af).all() and torch.isfinite(bf).all())}

def timed(fn,repeats=REPEATS):
    for _ in range(20): fn()
    torch.cuda.synchronize(); ev=[]; t0=time.perf_counter_ns()
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); ev.append((a,b))
    torch.cuda.synchronize(); t1=time.perf_counter_ns(); vals=[a.elapsed_time(b) for a,b in ev]
    return {'gpu_ms_mean':sum(vals)/len(vals),'gpu_ms_min':min(vals),'gpu_ms_max':max(vals),'wall_ms_mean':(t1-t0)/1e6/repeats,'repeats':repeats}

def build_attention_output(m):
    os.environ['DS41_MHC_COEFF_SINKHORN']='1'; os.environ['DS41_MHC_PROJECTION_RMS']='1'
    ids=torch.tensor(json.loads(m.PROMPT.read_text())['prompts']['arithmetic']['token_ids'],dtype=torch.long,device='cuda'); T=len(ids)
    linear=m.UnquantizedLinearMethod(); emb=m.bf16('token_embd')[ids]
    afn,asc,abase=m.f32('blk.0.hc_attn_fn'),m.f32('blk.0.hc_attn_scale'),m.f32('blk.0.hc_attn_base')
    residual=emb[:,None,:].expand(-1,m.HC,-1).contiguous(); afn_b=afn.reshape(24,m.HC,m.H).sum(1)
    _,_,x,_=m.mhc_pre_delayed_torch(residual,afn_b,asc,abase,m.EPS,m.HC_EPS,m.HC_EPS,2.0,m.SINK_ITERS,x=emb)
    xn=m.rms_candidate(x,m.bf16('blk.0.attn_norm'))
    fused=torch.cat((m.bf16('blk.0.attn_q_a'),m.bf16('blk.0.attn_kv')),dim=0).contiguous(); qkv=linear.apply(SimpleNamespace(weight=fused),xn)
    qra,kv=qkv.split([m.QRA,m.HD],dim=-1); qrn=m.rms_candidate(qra,m.bf16('blk.0.attn_q_a_norm')); kvn=m.rms_candidate(kv,m.bf16('blk.0.attn_kv_a_norm'))
    q=linear.apply(SimpleNamespace(weight=m.bf16('blk.0.attn_q_b')),qrn).reshape(T,m.HEADS,m.HD)
    pos=torch.arange(T,dtype=torch.int64,device='cuda'); cache=m.cos_sin_cache(T); qr=q.clone(); kr=kvn.clone(); m.vllm_ops.rotary_embedding(pos,qr,kr,m.HD,cache,False,rope_dim_offset=m.NOPE,inverse=False)
    kdq=m.cache_candidate_dequant(kr); ao=m.attention_candidate(qr,kdq,m.f32('blk.0.attn_sinks'))
    return ao,pos,cache

def main():
    assert torch.cuda.is_available(); m=load_l0(); ao,pos,cache=build_attention_output(m)
    woa=m.bf16('blk.0.attn_output_a').reshape(m.GROUPS,m.ORANK,m.HEADS*m.HD//m.GROUPS).contiguous()
    rows=[]
    for ti in range(ao.shape[0]-4,ao.shape[0]):
        o=ao[ti:ti+1].contiguous(); p=pos[ti:ti+1]
        inv=_fused_inverse_rope_gptj(o,p,cache,m.ROPE).view(1,m.GROUPS,-1)
        ref=torch.einsum('tgd,grd->tgr',inv,woa)
        cand=torch.bmm(inv.transpose(0,1).contiguous(),woa.transpose(1,2)).transpose(0,1)
        met=metric(cand,ref)
        if not met['finite'] or met['rel_l2']>REL_MAX or met['max_abs']>ABS_MAX: raise RuntimeError((ti,met))
        rows.append({'token_index':ti,**met})
    o=ao[-1:].contiguous(); p=pos[-1:]
    def ref_full():
        inv=_fused_inverse_rope_gptj(o,p,cache,m.ROPE).view(1,m.GROUPS,-1)
        return torch.einsum('tgd,grd->tgr',inv,woa)
    def cand_full():
        inv=_fused_inverse_rope_gptj(o,p,cache,m.ROPE).view(1,m.GROUPS,-1)
        return torch.bmm(inv.transpose(0,1).contiguous(),woa.transpose(1,2)).transpose(0,1)
    inv=_fused_inverse_rope_gptj(o,p,cache,m.ROPE).view(1,m.GROUPS,-1)
    def ref_proj(): return torch.einsum('tgd,grd->tgr',inv,woa)
    def cand_proj(): return torch.bmm(inv.transpose(0,1).contiguous(),woa.transpose(1,2)).transpose(0,1)
    bt=timed(ref_full); ct=timed(cand_full); bp=timed(ref_proj); cp=timed(cand_proj)
    out={'status':'PASS','gate':{'rel_l2_max':REL_MAX,'max_abs':ABS_MAX},'metrics':rows,
         'full_inverse_rope_plus_woa':{'baseline':bt,'candidate':ct,'gpu_speedup':bt['gpu_ms_mean']/ct['gpu_ms_mean'],'wall_speedup':bt['wall_ms_mean']/ct['wall_ms_mean']},
         'woa_projection_only':{'baseline':bp,'candidate':cp,'gpu_speedup':bp['gpu_ms_mean']/cp['gpu_ms_mean'],'wall_speedup':bp['wall_ms_mean']/cp['wall_ms_mean']},
         'shapes':{'o':list(o.shape),'inv_grouped':list(inv.shape),'wo_a':list(woa.shape)}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
