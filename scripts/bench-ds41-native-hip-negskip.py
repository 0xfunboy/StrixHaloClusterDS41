#!/usr/bin/env python3
"""Production-semantics M=1 A/B: Triton EP skip vs native HIP -1 skip."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch, gguf
from gguf.quants import dequantize
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, ApplyMoEActivationConfig, apply_moe_activation
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf_impl
from vllm_gguf_plugin import ops
ROOT=Path('/home/funboy/StrixHaloClusterDS41')
FIX=ROOT/'reports/DS41-Q2-001/perf/native-hip/layer0-moe-input.npz'
LIB=Path(os.environ.get('DS41_NATIVE_HIP_MOE_LIB','/home/funboy/models/ds41/native-hip-moe/runtime/_C_gguf.abi3.so'))
OUT=ROOT/'reports/DS41-Q2-001/perf/native-hip-negskip/production-semantics.json'
MODEL=Path(json.loads((ROOT/'runtime/ds41/artifact.json').read_text())['model_dir'])/'DSV41-mixedq2-00001-of-00005.gguf'
LIMIT=10.; REPEATS=50

def metric(a,b):
    af,bf=a.float(),b.float(); d=af-bf; rn=float(bf.norm())
    return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'rmse':float(torch.sqrt((d*d).mean())),'rel_l2':float(d.norm())/max(rn,1e-30),'a_norm':float(af.norm()),'b_norm':rn,'finite':bool(torch.isfinite(af).all() and torch.isfinite(bf).all())}

def act(inp):
    d=inp.shape[-1]//2; out=torch.empty(inp.shape[:-1]+(d,),dtype=inp.dtype,device=inp.device)
    apply_moe_activation(MoEActivation.SILU,out,inp,activation_config=ApplyMoEActivationConfig(clamp_limit=LIMIT)); return out

def native_prod(x,w13,w2,global_ids,weights,expert_map,q1,q2):
    # Same device-only mapping used by current EP fallback, but retain -1 so
    # native kernels can return before weight I/O for peer-owned routes.
    local_ids=expert_map[global_ids.to(torch.long)].to(torch.int32)
    k=global_ids.shape[1]
    gu=torch.ops._C_gguf.ggml_moe_a8_vec(x,w13,local_ids,k,q1,w13.shape[1],x.shape[0])
    a=act(gu)
    down=torch.ops._C_gguf.ggml_moe_a8_vec(a,w2,local_ids,1,q2,w2.shape[1],x.shape[0]*k)
    weighted=down.reshape(x.shape[0],k,w2.shape[1]).mul_(weights.view(x.shape[0],k,1))
    out=torch.empty_like(x); ops.moe_sum(weighted,out); return out

def triton_prod(x,w13,w2,global_ids,weights,expert_map,q1,q2):
    return _fused_moe_gguf_impl(x,w13,w2,weights,global_ids,q1,q2,'silu',LIMIT,expert_map)

def ref_local(x,global_ids,weights,owned,gt,ut,dt):
    out=torch.zeros_like(x,dtype=torch.float32)
    for slot,gid in enumerate(global_ids[0].tolist()):
        if gid not in owned: continue
        wg=dequantize(np.ascontiguousarray(gt.data[gid]),gt.tensor_type).astype(np.float32,copy=False)
        wu=dequantize(np.ascontiguousarray(ut.data[gid]),ut.tensor_type).astype(np.float32,copy=False)
        wd=dequantize(np.ascontiguousarray(dt.data[gid]),dt.tensor_type).astype(np.float32,copy=False)
        Wg=torch.from_numpy(wg).to('cuda'); Wu=torch.from_numpy(wu).to('cuda'); Wd=torch.from_numpy(wd).to('cuda')
        g=(x.float()@Wg.T).to(torch.bfloat16).float(); u=(x.float()@Wu.T).to(torch.bfloat16).float()
        g=torch.clamp(g,max=LIMIT); u=torch.clamp(u,-LIMIT,LIMIT); a=(g*torch.sigmoid(g)*u).to(torch.bfloat16)
        y=(a.float()@Wd.T).to(torch.bfloat16).float(); out.add_(y*weights[:,slot:slot+1])
        del Wg,Wu,Wd,wg,wu,wd,g,u,a,y
    return out.to(torch.bfloat16)

def timed(fn,*args):
    for _ in range(8): fn(*args)
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); t=time.perf_counter(); s.record(); last=None
    for _ in range(REPEATS): last=fn(*args)
    e.record(); e.synchronize(); wall=(time.perf_counter()-t)*1000/REPEATS
    return {'repeats':REPEATS,'gpu_ms_mean':float(s.elapsed_time(e))/REPEATS,'wall_ms_mean':wall},last

def main():
    assert os.environ.get('VLLM_GGUF_USE_CUDA','0')=='0'
    torch.ops.load_library(str(LIB))
    z=np.load(FIX); xall=torch.from_numpy(z['x_bf16_bits'].copy()).view(torch.bfloat16).to('cuda'); x=xall[-1:].contiguous()
    gids=torch.from_numpy(z['topk_ids'][-1:].copy()).to('cuda',dtype=torch.int32); weights=torch.from_numpy(z['topk_weights'][-1:].copy()).to('cuda')
    r=gguf.GGUFReader(str(MODEL)); by={t.name:t for t in r.tensors}; gt,ut,dt=by['blk.0.ffn_gate_exps'],by['blk.0.ffn_up_exps'],by['blk.0.ffn_down_exps']; q1,q2=int(gt.tensor_type),int(dt.tensor_type)
    report={'status':'RUNNING','native_lib':str(LIB),'fixture':str(FIX),'global_top6':gids.cpu().tolist()[0],'qtypes':[gt.tensor_type.name,dt.tensor_type.name],'ranks':{}}
    for rank,(lo,hi) in enumerate(((0,192),(192,384))):
        selected=sorted({int(v) for v in gids.cpu().flatten().tolist() if lo<=int(v)<hi})
        idx=np.asarray(selected,dtype=np.int64)
        gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).to('cuda'); up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).to('cuda'); down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).to('cuda'); w13=torch.cat((gate,up),dim=1).contiguous(); del gate,up
        emap=torch.full((384,),-1,dtype=torch.int32,device='cuda')
        for local,g in enumerate(selected): emap[g]=local
        tri=triton_prod(x,w13,down,gids,weights,emap,q1,q2); nat=native_prod(x,w13,down,gids,weights,emap,q1,q2); ref=ref_local(x,gids,weights,set(selected),gt,ut,dt)
        tt,tri2=timed(triton_prod,x,w13,down,gids,weights,emap,q1,q2); nt,nat2=timed(native_prod,x,w13,down,gids,weights,emap,q1,q2)
        local_ids=emap[gids.to(torch.long)].cpu().tolist()[0]
        report['ranks'][str(rank)]={'owned_selected':selected,'mapped_top6':local_ids,'triton_vs_reference':metric(tri,ref),'native_vs_reference':metric(nat,ref),'native_vs_triton':metric(nat,tri),'repeat_native_vs_triton':metric(nat2,tri2),'triton_timing':tt,'native_timing':nt,'speedup_gpu':tt['gpu_ms_mean']/nt['gpu_ms_mean'],'speedup_wall':tt['wall_ms_mean']/nt['wall_ms_mean']}
        del w13,down,emap,tri,nat,ref,tri2,nat2
        torch.cuda.empty_cache()
    report['status']='PASS' if all(v['native_vs_reference']['finite'] and v['native_vs_reference']['rel_l2']<0.08 for v in report['ranks'].values()) else 'FAIL'
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    if report['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
