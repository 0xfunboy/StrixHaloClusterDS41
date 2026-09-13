#!/usr/bin/env python3
"""Model-free DS41 wo_b M=1 LLMM1 gate on reconstructed layer0 inputs.

The canonical arithmetic prompt and DenseFix weights reconstruct four layer0
output-projection vectors. No attention fixture from attempt021 is used because
that attempt saved zero attention fixtures. M=1 is a shape, not a decode marker.
The candidate changes only the local RowParallelLinear GEMM. Manual shard sums
check reduction arithmetic; this script does not execute a distributed all-reduce.
Set DS41_WOB_LLMM1_OUT to a new report path to preserve previous raw results.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
REL_MAX = 5e-3
ABS_MAX = 1.25e-1
REPEATS = int(os.environ.get('DS41_WOB_LLMM1_REPEATS', '300'))


def load_l0():
    p = ROOT / 'scripts/test-ds41-layer0-complete-densefix.py'
    spec = importlib.util.spec_from_file_location('ds41_l0', p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float(), b.float(); d = af - bf
    rn = float(bf.norm())
    return {
        'max_abs': float(d.abs().max()),
        'mean_abs': float(d.abs().mean()),
        'rel_l2': float(d.norm()) / max(rn, 1e-30),
        'finite': bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
        'cand_norm': float(af.norm()),
        'ref_norm': rn,
        'exact_equal': bool(torch.equal(a, b)),
    }


def check_metric(name: str, value: dict) -> None:
    if not value['finite'] or value['rel_l2'] > REL_MAX or value['max_abs'] > ABS_MAX:
        raise RuntimeError(f'{name}: {value}')


def timed(fn, repeats: int) -> dict:
    for _ in range(20): fn()
    torch.cuda.synchronize()
    ev=[]; t0=time.perf_counter_ns()
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); ev.append((a,b))
    torch.cuda.synchronize(); t1=time.perf_counter_ns()
    vals=[a.elapsed_time(b) for a,b in ev]
    return {
        'repeats': repeats,
        'gpu_ms_mean': float(sum(vals)/len(vals)),
        'gpu_ms_min': float(min(vals)),
        'gpu_ms_max': float(max(vals)),
        'wall_ms_mean': (t1-t0)/1e6/repeats,
    }


def build_real_z(mod):
    os.environ['DS41_MHC_COEFF_SINKHORN']='1'
    os.environ['DS41_MHC_PROJECTION_RMS']='1'
    prompt=json.loads(mod.PROMPT.read_text())['prompts']['arithmetic']['token_ids']
    ids=torch.tensor(prompt,dtype=torch.long,device='cuda'); T=len(prompt)
    linear=mod.UnquantizedLinearMethod()
    emb=mod.bf16('token_embd')[ids]
    afn,asc,abase=mod.f32('blk.0.hc_attn_fn'),mod.f32('blk.0.hc_attn_scale'),mod.f32('blk.0.hc_attn_base')
    residual=emb[:,None,:].expand(-1,mod.HC,-1).contiguous()
    afn_b=afn.reshape(24,mod.HC,mod.H).sum(1)
    _,_,x,_=mod.mhc_pre_delayed_torch(residual,afn_b,asc,abase,mod.EPS,mod.HC_EPS,mod.HC_EPS,2.0,mod.SINK_ITERS,x=emb)
    an=mod.bf16('blk.0.attn_norm'); xn=mod.rms_candidate(x,an)
    wqa,wkv=mod.bf16('blk.0.attn_q_a'),mod.bf16('blk.0.attn_kv')
    fused=torch.cat((wqa,wkv),dim=0).contiguous(); qkv=linear.apply(SimpleNamespace(weight=fused),xn)
    qra,kv=qkv.split([mod.QRA,mod.HD],dim=-1)
    qn,kn=mod.bf16('blk.0.attn_q_a_norm'),mod.bf16('blk.0.attn_kv_a_norm')
    qrn,kvn=mod.rms_candidate(qra,qn),mod.rms_candidate(kv,kn)
    wqb=mod.bf16('blk.0.attn_q_b'); q=linear.apply(SimpleNamespace(weight=wqb),qrn).reshape(T,mod.HEADS,mod.HD)
    positions=torch.arange(T,dtype=torch.int64,device='cuda'); cache=mod.cos_sin_cache(T)
    qr=q.clone(); kr=kvn.clone(); mod.vllm_ops.rotary_embedding(positions,qr,kr,mod.HD,cache,False,rope_dim_offset=mod.NOPE,inverse=False)
    kdq=mod.cache_candidate_dequant(kr)
    sinks=mod.f32('blk.0.attn_sinks'); ao=mod.attention_candidate(qr,kdq,sinks)
    inv=mod._fused_inverse_rope_gptj(ao,positions,cache,mod.ROPE)
    woa=mod.bf16('blk.0.attn_output_a').reshape(mod.GROUPS,mod.ORANK,mod.HEADS*mod.HD//mod.GROUPS)
    z=torch.einsum('tgd,grd->tgr',inv.reshape(T,mod.GROUPS,-1),woa).flatten(1).contiguous()
    return z


def main():
    if not os.environ.get('DS41_WOB_LLMM1_OUT'):
        raise RuntimeError('Set DS41_WOB_LLMM1_OUT to a new report path.')
    out_path = Path(os.environ['DS41_WOB_LLMM1_OUT'])
    if out_path.exists():
        raise FileExistsError(f'Preserving existing raw report: {out_path}')
    assert torch.cuda.is_available() and torch.version.hip is not None
    assert REPEATS > 0
    mod=load_l0()
    z=build_real_z(mod)
    assert z.shape[0] >= 4 and z.shape[1] == 8192
    wob=mod.bf16('blk.0.attn_output_b').contiguous()
    assert tuple(wob.shape)==(5120,8192)
    linear = mod.UnquantizedLinearMethod()
    token_indices = list(range(z.shape[0] - 4, z.shape[0]))
    rows=[]; local_refs=[]; local_cands=[]
    for rank in (0,1):
        lo=rank*4096; hi=lo+4096
        w=wob[:,lo:hi].contiguous()
        layer = SimpleNamespace(weight=w)
        # Four reconstructed canonical vectors, evaluated as M=1; timing uses last.
        metrics=[]
        rank_refs=[]; rank_cands=[]
        for token_index in token_indices:
            x=z[token_index:token_index+1,lo:hi].contiguous()
            ref=linear.apply(layer,x,None)
            llmm1=mod.vllm_ops.LLMM1(w,x,4)
            ml=metric(llmm1,ref)
            baseline_equivalence=metric(ref,F.linear(x,w))
            check_metric(f'rank{rank} token{token_index} LLMM1 vs configured baseline',ml)
            check_metric(f'rank{rank} token{token_index} baseline vs F.linear',baseline_equivalence)
            metrics.append({
                'token_index':token_index, 'llmm1':ml,
                'baseline_vs_f_linear':baseline_equivalence,
            })
            rank_refs.append(ref); rank_cands.append(llmm1)
        x=z[-1:,lo:hi].contiguous()
        base_t=timed(lambda: linear.apply(layer,x,None),REPEATS)
        llmm1_t=timed(lambda: mod.vllm_ops.LLMM1(w,x,4),REPEATS)
        local_refs.append(torch.cat(rank_refs,dim=0))
        local_cands.append(torch.cat(rank_cands,dim=0))
        rows.append({
            'rank':rank,'x_shape':list(x.shape),'weight_shape':list(w.shape),'dtype':str(x.dtype),
            'metrics':metrics,'baseline':base_t,'llmm1':llmm1_t,
            'chosen':'LLMM1','candidate':llmm1_t,
            'gpu_speedup':base_t['gpu_ms_mean']/llmm1_t['gpu_ms_mean'],
            'wall_speedup':base_t['wall_ms_mean']/llmm1_t['wall_ms_mean'],
        })
    # Manual arithmetic on both rank shards, for every tested canonical vector.
    # A live TP run is still needed to validate distributed collective execution.
    ref_sum=(local_refs[0].float()+local_refs[1].float()).to(torch.bfloat16)
    cand_sum=(local_cands[0].float()+local_cands[1].float()).to(torch.bfloat16)
    tp_metric=metric(cand_sum,ref_sum)
    check_metric('LLMM1 shard sum vs configured baseline shard sum',tp_metric)
    tp_token_metrics=[]
    for index,token_index in enumerate(token_indices):
        value=metric(cand_sum[index],ref_sum[index])
        check_metric(f'token{token_index} LLMM1 shard sum',value)
        tp_token_metrics.append({'token_index':token_index,'metric':value})
    out={
        'status':'PASS','gate':{'rel_l2_max':REL_MAX,'max_abs':ABS_MAX},
        'source':'reconstructed canonical arithmetic layer0 output-projection inputs; DenseFix wo_b TP shards',
        'provenance':{
            'input_builder':'scripts/test-ds41-attn-wob-llmm1.py:build_real_z',
            'prompt_path':str(mod.PROMPT),'prompt_key':'arithmetic',
            'token_indices':token_indices,
            'attempt021_saved_attention_fixture_count':0,
            'attempt021_saved_attention_fixture_used':False,
            'live_decode_capture':False,
        },
        'baseline_method':'UnquantizedLinearMethod.apply(layer, x, None)',
        'baseline_gemm_impl':f'{linear._gemm_impl.__module__}.{linear._gemm_impl.__qualname__}',
        'candidate_method':'LLMM1(weight, x, 4)',
        'environment':{name:os.environ.get(name) for name in (
            'VLLM_ROCM_USE_SKINNY_GEMM','VLLM_ROCM_USE_AITER','VLLM_BATCH_INVARIANT',
        )},
        'rows':rows,'tp_sum_metric':tp_metric,'tp_sum_token_metrics':tp_token_metrics,
        'real_collective_tested':False,
        'scope':'M=1 shape, including possible one-token prefill or dummy calls',
    }
    out_path.parent.mkdir(parents=True,exist_ok=True)
    with out_path.open('x') as handle:
        handle.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
