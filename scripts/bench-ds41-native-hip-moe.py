#!/usr/bin/env python3
"""Compare DS41 Triton routed M=1 with isolated native HIP GGUF MMVQ.

Uses the real layer0 FFN input/router fixture captured from the canonical
DenseFix prompt.  Native ggml_moe_a8_vec includes its Q8_1 activation
quantization.  We compare complete local EP contributions through gate/up,
SwiGLU clamp=10, Q2_K down, route weights and sum.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch
import gguf
from gguf.quants import dequantize
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation, ApplyMoEActivationConfig, apply_moe_activation,
)
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf_impl
from vllm_gguf_plugin import ops

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
FIX=ROOT/'reports/DS41-Q2-001/perf/native-hip/layer0-moe-input.npz'
LIB=Path(os.environ.get('DS41_NATIVE_HIP_MOE_LIB','/home/funboy/models/ds41/native-hip-moe/runtime/_C_gguf.abi3.so'))
OUT=ROOT/'reports/DS41-Q2-001/perf/native-hip/native-vs-triton-real.json'
MODEL=Path(json.loads((ROOT/'runtime/ds41/artifact.json').read_text())['model_dir'])/'DSV41-mixedq2-00001-of-00005.gguf'
LIMIT=10.0
REPEATS=30


def metric(a:torch.Tensor,b:torch.Tensor):
    af=a.float(); bf=b.float(); d=af-bf; rn=float(bf.norm())
    return {
        'max_abs':float(d.abs().max()), 'mean_abs':float(d.abs().mean()),
        'rmse':float(torch.sqrt((d*d).mean())), 'rel_l2':float(d.norm())/max(rn,1e-30),
        'a_norm':float(af.norm()), 'b_norm':rn,
        'finite':bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
    }


def act_clamp(inp):
    d=inp.shape[-1]//2
    out=torch.empty(inp.shape[:-1]+(d,),dtype=inp.dtype,device=inp.device)
    apply_moe_activation(MoEActivation.SILU,out,inp,activation_config=ApplyMoEActivationConfig(clamp_limit=LIMIT))
    return out


def native_local(x,w13,w2,ids,weights,q1,q2):
    k=ids.shape[1]
    gu=torch.ops._C_gguf.ggml_moe_a8_vec(x,w13,ids,k,q1,w13.shape[1],x.shape[0])
    a=act_clamp(gu)
    down=torch.ops._C_gguf.ggml_moe_a8_vec(a,w2,ids,1,q2,w2.shape[1],x.shape[0]*k)
    weighted=down.reshape(x.shape[0],k,w2.shape[1]).mul_(weights.view(x.shape[0],k,1))
    out=torch.empty_like(x)
    ops.moe_sum(weighted,out)
    return out


def triton_local(x,w13,w2,ids,weights,q1,q2):
    return _fused_moe_gguf_impl(x,w13,w2,weights,ids,q1,q2,'silu',LIMIT,None)


def reference_local(x,weights,global_ids,gt,ut,dt):
    result=torch.zeros_like(x,dtype=torch.float32)
    for slot,gid in enumerate(global_ids):
        wg=dequantize(np.ascontiguousarray(gt.data[gid]),gt.tensor_type).astype(np.float32,copy=False)
        wu=dequantize(np.ascontiguousarray(ut.data[gid]),ut.tensor_type).astype(np.float32,copy=False)
        wd=dequantize(np.ascontiguousarray(dt.data[gid]),dt.tensor_type).astype(np.float32,copy=False)
        Wg=torch.from_numpy(wg).to('cuda'); Wu=torch.from_numpy(wu).to('cuda'); Wd=torch.from_numpy(wd).to('cuda')
        gate=(x.float()@Wg.T).to(torch.bfloat16).float()
        up=(x.float()@Wu.T).to(torch.bfloat16).float()
        gate=torch.clamp(gate,max=LIMIT); up=torch.clamp(up,min=-LIMIT,max=LIMIT)
        a=(gate*torch.sigmoid(gate)*up).to(torch.bfloat16)
        y=(a.float()@Wd.T).to(torch.bfloat16).float()
        result.add_(y*weights[:,slot:slot+1])
        del Wg,Wu,Wd,wg,wu,wd,gate,up,a,y
    return result.to(torch.bfloat16)


def time_fn(fn,*args,repeats=REPEATS):
    for _ in range(5): fn(*args)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    t0=time.perf_counter(); start.record()
    last=None
    for _ in range(repeats): last=fn(*args)
    end.record(); end.synchronize(); wall=time.perf_counter()-t0
    return {'repeats':repeats,'gpu_ms_mean':float(start.elapsed_time(end))/repeats,'wall_ms_mean':wall*1000/repeats}, last


def main():
    assert os.environ.get('VLLM_GGUF_USE_CUDA','0')=='0', 'Triton baseline must stay forced'
    assert torch.cuda.is_available() and LIB.is_file() and FIX.is_file()
    # Import plugin first with CUDA/native disabled, then register isolated native op.
    torch.ops.load_library(str(LIB))
    print('NATIVE_SCHEMA',torch.ops._C_gguf.ggml_moe_a8_vec.default._schema,flush=True)
    z=np.load(FIX)
    bits=z['x_bf16_bits']; xall=torch.from_numpy(bits.copy()).view(torch.bfloat16).to('cuda')
    ids_all=z['topk_ids']; weights_all=z['topk_weights']
    x=xall[-1:].contiguous(); gids=[int(v) for v in ids_all[-1]]; wglobal=weights_all[-1]
    reader=gguf.GGUFReader(str(MODEL))
    by={t.name:t for t in reader.tensors}
    gt,ut,dt=by['blk.0.ffn_gate_exps'],by['blk.0.ffn_up_exps'],by['blk.0.ffn_down_exps']
    q1,q2=int(gt.tensor_type),int(dt.tensor_type)
    assert gt.tensor_type.name=='IQ2_XXS' and dt.tensor_type.name=='Q2_K'
    report={'status':'RUNNING','schema':'ds41-native-hip-moe-v1','model':str(MODEL),'fixture':str(FIX),'native_lib':str(LIB),'x_norm':float(x.float().norm()),'global_top6':gids,'global_weights':[float(v) for v in wglobal],'qtypes':{'gate_up':gt.tensor_type.name,'down':dt.tensor_type.name},'ranks':{}}
    for rank,(lo,hi) in enumerate(((0,192),(192,384))):
        slots=[i for i,g in enumerate(gids) if lo<=g<hi]
        globals_local=[gids[i] for i in slots]
        if not slots: continue
        idx=np.asarray(globals_local,dtype=np.int64)
        gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).to('cuda')
        up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).to('cuda')
        down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).to('cuda')
        w13=torch.cat((gate,up),dim=1).contiguous(); del gate,up
        local_ids=torch.arange(len(slots),dtype=torch.int32,device='cuda').view(1,-1)
        local_weights=torch.tensor([float(wglobal[i]) for i in slots],dtype=torch.float32,device='cuda').view(1,-1)
        tri=triton_local(x,w13,down,local_ids,local_weights,q1,q2)
        nat=native_local(x,w13,down,local_ids,local_weights,q1,q2)
        ref=reference_local(x,local_weights,globals_local,gt,ut,dt)
        tri_t,tri2=time_fn(triton_local,x,w13,down,local_ids,local_weights,q1,q2)
        nat_t,nat2=time_fn(native_local,x,w13,down,local_ids,local_weights,q1,q2)
        row={
            'global_experts':globals_local,'route_weights':[float(wglobal[i]) for i in slots],
            'triton_vs_reference':metric(tri,ref),
            'native_vs_reference':metric(nat,ref),
            'native_vs_triton':metric(nat,tri),
            'repeat_native_vs_triton':metric(nat2,tri2),
            'triton_timing':tri_t,'native_timing':nat_t,
            'native_speedup_pct_gpu':(tri_t['gpu_ms_mean']/nat_t['gpu_ms_mean']-1)*100,
            'native_speedup_pct_wall':(tri_t['wall_ms_mean']/nat_t['wall_ms_mean']-1)*100,
        }
        report['ranks'][str(rank)]=row
        del w13,down,local_ids,local_weights,tri,nat,ref,tri2,nat2
        torch.cuda.empty_cache()
    # Native is a candidate only if finite and reasonably close to current/reference.
    for row in report['ranks'].values():
        if (not row['native_vs_reference']['finite'] or row['native_vs_reference']['rel_l2']>0.08 or row['native_vs_triton']['rel_l2']>0.08):
            report['status']='NUMERICAL_FAIL'; break
    else:
        report['status']='PASS'
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
