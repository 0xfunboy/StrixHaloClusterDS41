#!/usr/bin/env python3
from __future__ import annotations
import argparse, ctypes, json, os, statistics
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch, gguf
from _ds41_artifact import MODEL_DIR
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

REPEATS=3

def metric(a:torch.Tensor,b:torch.Tensor):
    af=a.float(); bf=b.float(); d=af-bf; n=float(bf.norm())
    return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'rel_l2':float(d.norm())/max(n,1e-30),'exact':bool(torch.equal(a,b))}

def event_ms(fn):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); y=fn(); e.record(); e.synchronize(); return y,float(s.elapsed_time(e))

def layer(w13,w2,emap):
    return SimpleNamespace(apply_router_weight_on_input=False,w13_weight=w13,w2_weight=w2,
        w13_weight_type=SimpleNamespace(weight_type=16),w2_weight_type=SimpleNamespace(weight_type=10),
        activation=SimpleNamespace(value='silu'),expert_map=emap)
def method(limit): return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=limit))

def ptr(t): return ctypes.c_void_p(t.data_ptr())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fixture',required=True); ap.add_argument('--lib',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    fix=torch.load(a.fixture,weights_only=False)
    assert fix['expert_map'].dtype==torch.int32
    rank=int(fix['rank']); li=int(fix['layer_index']); T=int(fix['tokens']); H=int(fix['x'].shape[1]); MID=2304; GEXP=384
    em_cpu=fix['expert_map'].to(torch.int64); local_by_id=sorted((int(l),g) for g,l in enumerate(em_cpu.tolist()) if int(l)>=0)
    assert [l for l,_ in local_by_id]==list(range(len(local_by_id))); globals_idx=np.asarray([g for _,g in local_by_id],dtype=np.int64); LEXP=len(globals_idx)
    readers=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]; by={t.name:t for r in readers for t in r.tensors}
    gt,ut,dt=(by[f'blk.{li}.ffn_gate_exps'],by[f'blk.{li}.ffn_up_exps'],by[f'blk.{li}.ffn_down_exps'])
    # Local expert weights in captured expert-map order. Raw GGUF bytes are kept unchanged.
    gate=torch.from_numpy(np.ascontiguousarray(gt.data[globals_idx])).to('cuda'); up=torch.from_numpy(np.ascontiguousarray(ut.data[globals_idx])).to('cuda'); down=torch.from_numpy(np.ascontiguousarray(dt.data[globals_idx])).to('cuda')
    x=fix['x'].to('cuda').contiguous(); gids=fix['topk_ids'].to(device='cuda',dtype=torch.int32).contiguous(); rw=fix['topk_weights'].to(device='cuda',dtype=torch.float32).contiguous(); em=fix['expert_map'].to(device='cuda',dtype=torch.int32).contiguous(); limit=float(fix['swiglu_limit'])
    localmask=em_cpu[fix['topk_ids'].to(torch.int64)]>=0; local_routes=int(localmask.sum()); remote_routes=int(localmask.numel()-local_routes)
    # Baseline DS41 Triton: same current EP skip-remote path.
    os.environ['DS41_EP_SKIP_REMOTE']='1'; os.environ['DS41_NATIVE_HIP_MOE']='1'; os.environ['DS41_NATIVE_HIP_MOE_ROWWISE']='1'; os.environ['DS41_MOE_PREFILL_BLOCK_M']='4'
    w13=torch.cat((gate,up),dim=1); L=layer(w13,down,em); M=method(limit)
    GGUFMoEMethod.apply(M,L,x,rw,gids,None,None); torch.cuda.synchronize(); btimes=[]; bout=None
    for _ in range(REPEATS): bout,ms=event_ms(lambda:GGUFMoEMethod.apply(M,L,x,rw,gids,None,None)); btimes.append(ms)
    bout=bout.detach().clone(); del w13,L,M; torch.cuda.empty_cache()
    # DS4 complete hot component scratch. Allocations/weight upload are one-time and excluded; preparation/conversions are timed.
    lib=ctypes.CDLL(a.lib,mode=ctypes.RTLD_GLOBAL); lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int
    rc=lib.ds4_mmq_init(0); assert rc==0,rc
    fn=lib.ds41_ds4_moe_full
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
                 ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,
                 ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    fn.restype=ctypes.c_int
    local_ids=torch.empty((T,6),device='cuda',dtype=torch.int32); xf=torch.empty((T,H),device='cuda',dtype=torch.float32)
    gateo=torch.empty((T*6,MID),device='cuda',dtype=torch.float32); upo=torch.empty_like(gateo); mid=torch.empty_like(gateo)
    routes=torch.empty((T*6,H),device='cuda',dtype=torch.float32); outf=torch.empty((T,H),device='cuda',dtype=torch.float32); outbf=torch.empty((T,H),device='cuda',dtype=torch.bfloat16)
    stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    def call():
        r=fn(ptr(x),ptr(gids),ptr(rw),ptr(em),ptr(gate),ptr(up),ptr(down),T,H,MID,GEXP,LEXP,ctypes.c_float(limit),ptr(local_ids),ptr(xf),ptr(gateo),ptr(upo),ptr(mid),ptr(routes),ptr(outf),stream)
        if r: raise RuntimeError(f'ds4 bridge rc={r}')
        outbf.copy_(outf)
        return outbf
    call(); torch.cuda.synchronize(); ctimes=[]; cout=None; torch.cuda.reset_peak_memory_stats(); base_mem=torch.cuda.memory_allocated()
    for _ in range(REPEATS): cout,ms=event_ms(call); ctimes.append(ms)
    cout=cout.detach().clone(); peak=torch.cuda.max_memory_allocated()
    # Verify bridge route map implements EP2 exactly.
    expected_local=em[gids.to(torch.long)]; map_exact=bool(torch.equal(local_ids,expected_local))
    result={'schema':'ds41-ds4-prefill-fixture-v1','status':'PASS' if map_exact and torch.isfinite(cout).all() else 'FAIL',
      'fixture':a.fixture,'rank':rank,'chunk_index':int(fix['chunk_index']),'layer_index':li,'tokens':T,'hidden':H,'mid_dim':MID,
      'local_experts':LEXP,'global_experts':GEXP,'local_routes':local_routes,'remote_routes':remote_routes,
      'precision_contract':{'ds41_input':'BF16','ds41_intermediate_boundaries':'BF16-oriented Triton GGUF path','ds4_input_boundary':'BF16->F32 timed, then DS4 internal Q8_1 activation quantization','ds4_gate_up_mid':'F32 materialized','ds4_down_output':'F32 sum then timed F32->BF16 cast'},
      'ep2_map_exact':map_exact,
      'baseline_ds41':{'samples_ms':btimes,'median_ms':statistics.median(btimes),'mean_ms':statistics.mean(btimes)},
      'candidate_ds4_raw_mmq':{'samples_ms':ctimes,'median_ms':statistics.median(ctimes),'mean_ms':statistics.mean(ctimes),'peak_delta_bytes':peak-base_mem},
      'speedup_ratio':statistics.median(btimes)/statistics.median(ctimes),'time_reduction_pct':(1-statistics.median(ctimes)/statistics.median(btimes))*100,
      'candidate_vs_ds41_output':metric(cout,bout)}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
