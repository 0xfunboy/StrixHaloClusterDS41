#!/usr/bin/env python3
"""Progressive real-prompt layer0 correctness gate on the repaired DenseFix artifact.

Independent reference boundary:
- BF16/F32 tensors are read directly from the repaired GGUF.
- Dense/HC/RMS/RoPE/attention/reference MoE math is explicit PyTorch/NumPy.
- Routed IQ2_XXS/Q2_K experts are independently dequantized with pinned gguf-py.
Candidate boundary:
- vLLM ROCm RMS/rotary/sparse-attention and GGUF routed-MoE kernels.

The script stops on a material boundary failure.  It never loads the full model.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch
from gguf.quants import dequantize

from vllm import _custom_ops as vllm_ops
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_delayed_torch
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _fused_inverse_rope_gptj,
    rocm_sparse_attn_prefill,
)
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
ART = json.loads((ROOT / 'runtime/ds41/artifact.json').read_text())
MODEL = Path(ART['model_dir'])
PROMPT = ROOT / 'reports/DS41-Q2-001/attempt011/prompt-tokens.json'
OUT = Path(os.environ.get('DS41_LAYER0_OUT', str(ROOT / 'reports/DS41-Q2-001/densefix/layer0-complete-node01.json')))
CAND_NPY = Path(os.environ.get('DS41_LAYER0_CAND_NPY', str(ROOT / 'reports/DS41-Q2-001/densefix/layer0-final-candidate.npy')))
REF_NPY = Path(os.environ.get('DS41_LAYER0_REF_NPY', str(ROOT / 'reports/DS41-Q2-001/densefix/layer0-final-reference.npy')))
CFG = json.loads((MODEL / 'config.json').read_text())['text_config']

H = int(CFG['hidden_size']); HC = int(CFG['hc_mult'])
HEADS = int(CFG['num_attention_heads']); HD = int(CFG['head_dim'])
ROPE = int(CFG['qk_rope_head_dim']); NOPE = HD - ROPE
QRA = int(CFG['q_lora_rank']); GROUPS = int(CFG['o_groups']); ORANK = int(CFG['o_lora_rank'])
TOPK = int(CFG['num_experts_per_tok']); LIMIT = float(CFG['swiglu_limit'])
EPS = float(CFG['rms_norm_eps']); HC_EPS = float(CFG['hc_eps']); SINK_ITERS = int(CFG['hc_sinkhorn_iters'])
SCALE = HD ** -0.5
assert CFG['compress_ratios'][0] == 0
assert (H, HC, HEADS, HD, ROPE, QRA, GROUPS, ORANK) == (5120,4,64,512,64,1280,8,1024)

readers = [gguf.GGUFReader(str(p)) for p in sorted(MODEL.glob('*.gguf'))]
by_name = {t.name: t for r in readers for t in r.tensors}


def tensor(name: str):
    return by_name[name]


def bf16(name: str, device: str = 'cuda') -> torch.Tensor:
    t=tensor(name); assert t.tensor_type.name == 'BF16'
    a=t.data.view(np.uint16).reshape(*(int(x) for x in reversed(t.shape))).copy()
    return torch.from_numpy(a).view(torch.bfloat16).to(device)


def f32(name: str, device: str = 'cuda') -> torch.Tensor:
    t=tensor(name); assert t.tensor_type.name == 'F32'
    a=np.asarray(t.data).view(np.float32).reshape(*(int(x) for x in reversed(t.shape))).copy()
    return torch.from_numpy(a).to(device)


def metric(cand: torch.Tensor, ref: torch.Tensor) -> dict:
    c=cand.float(); r=ref.float(); d=c-r; rn=float(r.norm())
    return {
        'shape': list(c.shape), 'max_abs': float(d.abs().max()),
        'mean_abs': float(d.abs().mean()), 'rmse': float(torch.sqrt((d*d).mean())),
        'rel_l2': float(d.norm())/max(rn,1e-30), 'ref_norm': rn,
        'cand_norm': float(c.norm()), 'cand_finite': bool(torch.isfinite(c).all()),
        'ref_finite': bool(torch.isfinite(r).all()),
    }


def assert_metric(report: dict, name: str, cand: torch.Tensor, ref: torch.Tensor, *, rel: float, max_abs: float) -> None:
    m=metric(cand,ref); report['boundaries'][name]=m
    if not m['cand_finite'] or not m['ref_finite'] or m['rel_l2']>rel or m['max_abs']>max_abs:
        report['status']='FAIL_MATERIAL'; report['first_material_mismatch']=name
        raise RuntimeError(f"material mismatch {name}: {m}")


def rms_ref(x,w):
    xf=x.float(); return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)*w.float()).to(torch.bfloat16)


def rms_candidate(x,w):
    out=torch.empty_like(x); vllm_ops.rms_norm(out,x,w,EPS); return out


def hc_mix_ref(h, fn, scale, base):
    xf=h.float().flatten(1)
    m=(xf@fn.float().T)*torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)
    pre=torch.sigmoid(m[:,:HC]*scale[0]+base[:HC])+HC_EPS
    post=2*torch.sigmoid(m[:,HC:2*HC]*scale[1]+base[HC:2*HC])
    comb=(m[:,2*HC:]*scale[2]+base[2*HC:]).reshape(-1,HC,HC)
    comb=torch.softmax(comb,dim=-1)+HC_EPS
    comb=comb/(comb.sum(-2,keepdim=True)+HC_EPS)
    for _ in range(SINK_ITERS-1):
        comb=comb/(comb.sum(-1,keepdim=True)+HC_EPS)
        comb=comb/(comb.sum(-2,keepdim=True)+HC_EPS)
    return pre, post[...,None], comb


def hc_post_ref(x,residual,post,comb):
    return (post.float()*x.float()[:,None,:] + torch.einsum('tij,tid->tjd',comb.float(),residual.float())).to(torch.bfloat16)


def cos_sin_cache(n: int, device='cuda'):
    inv=1.0/(float(CFG['rope_theta'])**(torch.arange(0,ROPE,2,dtype=torch.float32,device=device)/ROPE))
    pos=torch.arange(n,dtype=torch.float32,device=device)
    freqs=pos[:,None]*inv[None,:]
    return torch.cat([freqs.cos(),freqs.sin()],dim=-1)


def rope_ref(x: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor, inverse=False):
    out=x.float().clone(); tail=out[...,NOPE:]
    half=ROPE//2; cs=cache[positions]; cos=cs[:,:half]; sin=cs[:,half:]
    if inverse: sin=-sin
    while cos.ndim < tail.ndim:
        cos=cos.unsqueeze(1); sin=sin.unsqueeze(1)
    even=tail[...,0::2]; odd=tail[...,1::2]
    rotated=torch.stack((even*cos-odd*sin, even*sin+odd*cos),dim=-1).flatten(-2)
    out[...,NOPE:]=rotated
    return out.to(torch.bfloat16)


def cache_ref_dequant(kv_bf16: torch.Tensor) -> torch.Tensor:
    x=kv_bf16.detach().cpu().float(); t=x.shape[0]
    nope=x[:,:NOPE].reshape(t,7,64)
    absmax=nope.abs().amax(-1).clamp_min(1e-4)
    exponent=torch.ceil(torch.log2(absmax/448.0))
    scale=torch.exp2(exponent)
    q=(nope/scale[...,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    dq=(q.float()*scale[...,None]).reshape(t,NOPE)
    rope=kv_bf16.detach().cpu()[:,NOPE:].to(torch.bfloat16).float()
    return torch.cat((dq,rope),dim=-1).to(torch.bfloat16).to('cuda')


def cache_candidate_dequant(kv_bf16: torch.Tensor) -> torch.Tensor:
    block=32; token_data=576; scale_bytes=8
    cache=torch.zeros((1,block*token_data+block*scale_bytes),dtype=torch.uint8,device='cuda')
    slots=torch.arange(kv_bf16.shape[0],dtype=torch.int64,device='cuda')
    quantize_and_insert_k_cache(kv_bf16,cache,slots,block_size=block,is_ue8m0=True,use_fnuz=False)
    out=torch.empty((1,kv_bf16.shape[0],HD),dtype=torch.bfloat16,device='cuda')
    seq=torch.tensor([kv_bf16.shape[0]],dtype=torch.int32,device='cuda')
    gather=seq.clone(); table=torch.tensor([[0]],dtype=torch.int32,device='cuda')
    dequantize_and_gather_k_cache(out,cache,seq,gather,table,block,0,use_fnuz=False)
    return out[0]


def attention_ref(q,kv,sinks):
    T=q.shape[0]; qf=q.float(); kvf=kv.float(); out=torch.empty_like(qf)
    for t in range(T):
        keys=kvf[:t+1]
        scores=torch.einsum('hd,kd->hk',qf[t],keys)*SCALE
        full=torch.cat((scores,sinks.float()[:,None]),dim=-1)
        prob=torch.softmax(full,dim=-1)[...,:-1]
        out[t]=torch.einsum('hk,kd->hd',prob,keys)
    return out.to(torch.bfloat16)


def attention_candidate(q,kv,sinks):
    T=q.shape[0]; width=T
    indices=torch.full((T,width),-1,dtype=torch.int32,device='cuda')
    for t in range(T): indices[t,:t+1]=torch.arange(t+1,dtype=torch.int32,device='cuda')
    lens=torch.arange(1,T+1,dtype=torch.int32,device='cuda')
    out=torch.empty_like(q)
    rocm_sparse_attn_prefill(q=q,kv=kv[:,None,:],indices=indices,topk_length=lens,scale=SCALE,
        head_dim=HD,nope_head_dim=NOPE,rope_head_dim=ROPE,attn_sink=sinks,output=out)
    return out


def moe_layer(w13,w2):
    return SimpleNamespace(
        apply_router_weight_on_input=False,w13_weight=w13,w2_weight=w2,
        w13_weight_type=SimpleNamespace(weight_type=int(gguf.GGMLQuantizationType.IQ2_XXS)),
        w2_weight_type=SimpleNamespace(weight_type=int(gguf.GGMLQuantizationType.Q2_K)),
        activation=SimpleNamespace(value='silu'),expert_map=None,
    )


def moe_method():
    return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=LIMIT))


def routed_reference(x: torch.Tensor, topw: torch.Tensor, topids: torch.Tensor, gate_t, up_t, down_t) -> torch.Tensor:
    result=torch.zeros_like(x,dtype=torch.float32)
    unique=sorted(set(int(v) for v in topids.detach().cpu().flatten().tolist()))
    for n,e in enumerate(unique,1):
        pairs=(topids==e).nonzero(as_tuple=False)
        tids=pairs[:,0]; slots=pairs[:,1]
        rawg=np.ascontiguousarray(gate_t.data[e]); rawu=np.ascontiguousarray(up_t.data[e]); rawd=np.ascontiguousarray(down_t.data[e])
        wg=torch.from_numpy(dequantize(rawg,gate_t.tensor_type).astype(np.float32,copy=False)).to('cuda')
        wu=torch.from_numpy(dequantize(rawu,up_t.tensor_type).astype(np.float32,copy=False)).to('cuda')
        wd=torch.from_numpy(dequantize(rawd,down_t.tensor_type).astype(np.float32,copy=False)).to('cuda')
        xs=x[tids].float()
        ga=(xs@wg.T).to(torch.bfloat16).float(); up=(xs@wu.T).to(torch.bfloat16).float()
        ga=torch.clamp(ga,max=LIMIT); up=torch.clamp(up,-LIMIT,LIMIT)
        act=(ga*torch.sigmoid(ga)*up).to(torch.bfloat16)
        y=(act.float()@wd.T).to(torch.bfloat16).float()
        weights=topw[tids,slots].float()[:,None]
        result.index_add_(0,tids,y*weights)
        del wg,wu,wd,rawg,rawu,rawd,ga,up,act,y
        if n==1 or n%16==0 or n==len(unique): print(f"L0_ROUTE_REF experts={n}/{len(unique)}",flush=True)
    return result.to(torch.bfloat16)


def main():
    assert torch.cuda.is_available()
    prompt=json.loads(PROMPT.read_text())['prompts']['arithmetic']['token_ids']
    ids=torch.tensor(prompt,dtype=torch.long,device='cuda'); T=len(prompt)
    report={'status':'RUNNING','artifact':ART['name'],'model_dir':str(MODEL),'prompt_ids':prompt,'boundaries':{}}
    linear=UnquantizedLinearMethod()

    emb=bf16('token_embd')[ids]
    afn,asc,abase=f32('blk.0.hc_attn_fn'),f32('blk.0.hc_attn_scale'),f32('blk.0.hc_attn_base')
    residual=emb[:,None,:].expand(-1,HC,-1).contiguous()
    afn_b=afn.reshape(24,HC,H).sum(1)
    apost_c,acomb_c,x_c,apre_c=mhc_pre_delayed_torch(residual,afn_b,asc,abase,EPS,HC_EPS,HC_EPS,2.0,SINK_ITERS,x=emb)
    apre_r,apost_r,acomb_r=hc_mix_ref(residual,afn,asc,abase); x_r=emb
    assert_metric(report,'hc_attn_x',x_c,x_r,rel=2e-5,max_abs=2e-5)
    assert_metric(report,'hc_attn_pre',apre_c,apre_r,rel=1e-5,max_abs=1e-5)
    assert_metric(report,'hc_attn_post',apost_c,apost_r,rel=1e-5,max_abs=1e-5)
    assert_metric(report,'hc_attn_comb',acomb_c,acomb_r,rel=1e-5,max_abs=1e-5)

    an=bf16('blk.0.attn_norm'); xn_c=rms_candidate(x_c,an); xn_r=rms_ref(x_r,an)
    assert_metric(report,'attn_norm',xn_c,xn_r,rel=1e-5,max_abs=1e-5)
    wqa,wkv=bf16('blk.0.attn_q_a'),bf16('blk.0.attn_kv')
    fused=torch.cat((wqa,wkv),dim=0).contiguous(); qkv_c=linear.apply(SimpleNamespace(weight=fused),xn_c)
    qra_c,kv_c=qkv_c.split([QRA,HD],dim=-1)
    qra_r=(xn_r.float()@wqa.float().T).to(torch.bfloat16); kv_r=(xn_r.float()@wkv.float().T).to(torch.bfloat16)
    assert_metric(report,'q_a_projection',qra_c,qra_r,rel=2e-3,max_abs=2e-2)
    assert_metric(report,'kv_projection',kv_c,kv_r,rel=2e-3,max_abs=2e-2)
    qn,kn=bf16('blk.0.attn_q_a_norm'),bf16('blk.0.attn_kv_a_norm')
    qrn_c,kvn_c=rms_candidate(qra_c,qn),rms_candidate(kv_c,kn)
    qrn_r,kvn_r=rms_ref(qra_r,qn),rms_ref(kv_r,kn)
    assert_metric(report,'q_norm',qrn_c,qrn_r,rel=2e-3,max_abs=2e-2)
    assert_metric(report,'kv_norm',kvn_c,kvn_r,rel=2e-3,max_abs=2e-2)
    wqb=bf16('blk.0.attn_q_b'); q_c=linear.apply(SimpleNamespace(weight=wqb),qrn_c).reshape(T,HEADS,HD)
    q_r=(qrn_r.float()@wqb.float().T).to(torch.bfloat16).reshape(T,HEADS,HD)
    assert_metric(report,'q_b_projection',q_c,q_r,rel=2e-3,max_abs=2e-1)

    positions=torch.arange(T,dtype=torch.int64,device='cuda'); cache=cos_sin_cache(T)
    qr_c=q_c.clone(); kr_c=kvn_c.clone(); vllm_ops.rotary_embedding(positions,qr_c,kr_c,HD,cache,False,rope_dim_offset=NOPE,inverse=False)
    qr_r=rope_ref(q_r,positions,cache); kr_r=rope_ref(kvn_r,positions,cache)
    assert_metric(report,'q_rope',qr_c,qr_r,rel=3e-3,max_abs=2e-1)
    assert_metric(report,'kv_rope',kr_c,kr_r,rel=3e-3,max_abs=2e-1)
    kdq_c=cache_candidate_dequant(kr_c); kdq_r=cache_ref_dequant(kr_r)
    assert_metric(report,'fp8_ds_mla_cache',kdq_c,kdq_r,rel=1e-2,max_abs=8e-2)

    sinks=f32('blk.0.attn_sinks'); ao_c=attention_candidate(qr_c,kdq_c,sinks); ao_r=attention_ref(qr_r,kdq_r,sinks)
    assert_metric(report,'swa_attention_sinks',ao_c,ao_r,rel=3e-2,max_abs=1.5e-1)

    inv_c=_fused_inverse_rope_gptj(ao_c,positions,cache,ROPE)
    inv_r=rope_ref(ao_r,positions,cache,inverse=True)
    assert_metric(report,'inverse_rope',inv_c,inv_r,rel=3e-2,max_abs=1.5e-1)
    woa=bf16('blk.0.attn_output_a').reshape(GROUPS,ORANK,HEADS*HD//GROUPS)
    zc=torch.einsum('tgd,grd->tgr',inv_c.reshape(T,GROUPS,-1),woa)
    zr=torch.einsum('tgd,grd->tgr',inv_r.float().reshape(T,GROUPS,-1),woa.float()).to(torch.bfloat16)
    assert_metric(report,'wo_a',zc,zr,rel=4e-2,max_abs=2e-1)
    wob=bf16('blk.0.attn_output_b'); attn_c=linear.apply(SimpleNamespace(weight=wob),zc.flatten(1))
    attn_r=(zr.float().flatten(1)@wob.float().T).to(torch.bfloat16)
    assert_metric(report,'wo_b',attn_c,attn_r,rel=4e-2,max_abs=3e-1)

    h_attn_c=mhc_post_torch(attn_c,residual,apost_c,acomb_c); h_attn_r=hc_post_ref(attn_r,residual,apost_r,acomb_r)
    assert_metric(report,'attn_hc_residual',h_attn_c,h_attn_r,rel=4e-2,max_abs=3e-1)

    ffn_fn,ffn_sc,ffn_base=f32('blk.0.hc_ffn_fn'),f32('blk.0.hc_ffn_scale'),f32('blk.0.hc_ffn_base')
    fpost_c,fcomb_c,fx_c,fpre_c=mhc_pre_delayed_torch(h_attn_c,ffn_fn,ffn_sc,ffn_base,EPS,HC_EPS,HC_EPS,2.0,SINK_ITERS,pre_mix=apre_c)
    fpre_r,fpost_r,fcomb_r=hc_mix_ref(h_attn_r,ffn_fn,ffn_sc,ffn_base)
    fx_r=(h_attn_r.float()*apre_r[...,None]).sum(1).to(torch.bfloat16)
    assert_metric(report,'ffn_hc_x',fx_c,fx_r,rel=5e-2,max_abs=3e-1)
    assert_metric(report,'ffn_hc_pre',fpre_c,fpre_r,rel=5e-2,max_abs=5e-2)
    fnorm=bf16('blk.0.ffn_norm'); fn_c=rms_candidate(fx_c,fnorm); fn_r=rms_ref(fx_r,fnorm)
    assert_metric(report,'ffn_norm',fn_c,fn_r,rel=5e-2,max_abs=3e-1)

    gatew=bf16('blk.0.ffn_gate_inp'); bias=f32('blk.0.exp_probs_b')
    logits=fn_r.float()@gatew.float().T; scores=torch.sqrt(torch.nn.functional.softplus(logits))
    topids64=torch.topk(scores+bias,TOPK,dim=-1).indices
    selected=torch.gather(scores,1,topids64); selected=selected/(selected.sum(-1,keepdim=True)+1e-20)*float(CFG['routed_scaling_factor'])
    # The real V4.1 router hands int32 expert IDs to the GGUF MoE kernels.
    topids=topids64.to(torch.int32)
    report['router']={'topk_ids_last_token':topids[-1].tolist(),'topk_weights_last_token':[float(x) for x in selected[-1]],'weight_sums':[float(x) for x in selected.sum(-1)]}

    gt,ut,dt=tensor('blk.0.ffn_gate_exps'),tensor('blk.0.ffn_up_exps'),tensor('blk.0.ffn_down_exps')

    def ep_partition(start: int, end: int) -> torch.Tensor:
        selected_ids=sorted({int(v) for v in topids.detach().cpu().flatten().tolist() if start <= int(v) < end})
        if not selected_ids:
            return torch.zeros_like(fn_c)
        idx=np.asarray(selected_ids,dtype=np.int64)
        wg=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).to('cuda')
        wu=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).to('cuda')
        wd=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).to('cuda')
        w13=torch.cat((wg,wu),dim=1); del wg,wu
        expert_map=torch.full((384,),-1,dtype=torch.int32,device='cuda')
        for local_id,global_id in enumerate(selected_ids): expert_map[global_id]=local_id
        layer=moe_layer(w13,wd); layer.expert_map=expert_map
        out=GGUFMoEMethod.apply(moe_method(),layer,fn_c,selected,topids,None,None)
        report.setdefault('ep_partitions',[]).append({'range':[start,end],'selected_global_ids':selected_ids,'local_experts':len(selected_ids)})
        del w13,wd,expert_map,layer
        return out

    routed_c=(ep_partition(0,192).float()+ep_partition(192,384).float()).to(torch.bfloat16)
    routed_r=routed_reference(fn_r,selected,topids,gt,ut,dt)
    assert_metric(report,'routed_moe',routed_c,routed_r,rel=8e-2,max_abs=2.0)

    sw1,sw3,sw2=bf16('blk.0.ffn_gate_shexp'),bf16('blk.0.ffn_up_shexp'),bf16('blk.0.ffn_down_shexp')
    sg_c=linear.apply(SimpleNamespace(weight=sw1),fn_c); su_c=linear.apply(SimpleNamespace(weight=sw3),fn_c)
    spre_c=torch.cat((sg_c,su_c),dim=-1); sact_c=torch.empty_like(sg_c)
    apply_moe_activation(MoEActivation.SILU,sact_c,spre_c,activation_config=ApplyMoEActivationConfig(clamp_limit=LIMIT))
    shared_c=linear.apply(SimpleNamespace(weight=sw2),sact_c)
    sg=(fn_r.float()@sw1.float().T).to(torch.bfloat16).float(); su=(fn_r.float()@sw3.float().T).to(torch.bfloat16).float()
    sg=torch.clamp(sg,max=LIMIT); su=torch.clamp(su,-LIMIT,LIMIT); sact=(sg*torch.sigmoid(sg)*su).to(torch.bfloat16)
    shared_r=(sact.float()@sw2.float().T).to(torch.bfloat16)
    assert_metric(report,'shared_moe',shared_c,shared_r,rel=5e-2,max_abs=5e-1)
    moe_c=(routed_c.float()+shared_c.float()).to(torch.bfloat16); moe_r=(routed_r.float()+shared_r.float()).to(torch.bfloat16)
    assert_metric(report,'moe_total',moe_c,moe_r,rel=8e-2,max_abs=2.0)

    final_c=mhc_post_torch(moe_c,h_attn_c,fpost_c,fcomb_c); final_r=hc_post_ref(moe_r,h_attn_r,fpost_r,fcomb_r)
    assert_metric(report,'layer0_final_hc',final_c,final_r,rel=8e-2,max_abs=2.0)
    report['status']='PASS'; report['first_material_mismatch']=None
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n')
    np.save(CAND_NPY,final_c.detach().cpu().view(torch.uint16).numpy())
    np.save(REF_NPY,final_r.detach().cpu().view(torch.uint16).numpy())
    print(json.dumps(report,indent=2)); print(f"L0_COMPLETE=PASS out={OUT}")


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        # Preserve the first failing boundary even if execution stops.
        if 'report' in globals():
            report['error']=f"{type(exc).__name__}: {exc}"; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n')
        raise
