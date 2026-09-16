#!/usr/bin/env python3
from __future__ import annotations
import argparse, ctypes, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, gguf
from gguf.quants import dequantize
from _ds41_artifact import MODEL_DIR

G_STAGE=.04; G_FINAL=.05; G_ROW95=.08; G_SELF=1e-5

def round_away(x):
    return np.where(x>=0,np.floor(x+0.5),np.ceil(x-0.5))
def q8_emulate(x:np.ndarray, block:int)->np.ndarray:
    assert x.ndim==1 and x.size%block==0
    z=x.astype(np.float32,copy=False).reshape(-1,block)
    a=np.max(np.abs(z),axis=1,keepdims=True)
    d=a/np.float32(127.0)
    scale=np.where(a>0,np.float32(127.0)/a,np.float32(0.0))
    q=round_away(z*scale).clip(-127,127).astype(np.int8)
    return (q.astype(np.float32)*d).reshape(-1)
def metric(a,b):
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32); d=a-b
    n=float(np.linalg.norm(b)); ma=float(np.max(np.abs(d))) if d.size else 0.
    scale=float(np.max(np.abs(b))) if b.size else 0.; idx=int(np.argmax(np.abs(d))) if d.size else -1
    return {'rel_l2':float(np.linalg.norm(d))/max(n,1e-30),'max_abs':ma,'reference_abs_max':scale,'max_abs_over_scale':ma/max(scale,1e-30),'mean_abs':float(np.mean(np.abs(d))) if d.size else 0.,'worst_flat_index':idx}
def silu_weight(g,u,w,clamp):
    g=np.minimum(g,clamp) if clamp>1e-6 else g
    u=np.clip(u,-clamp,clamp) if clamp>1e-6 else u
    g=np.where(np.isfinite(g),g,0).astype(np.float32); u=np.where(np.isfinite(u),u,0).astype(np.float32)
    return ((g/(1+np.exp(-g,dtype=np.float32)))*u*np.float32(w)).astype(np.float32)
def ptr(t): return ctypes.c_void_p(t.data_ptr())

def readers():
    rs=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]
    return {t.name:t for r in rs for t in r.tensors}
def load_fix(path,holdout_rank=None,expert_map_fixture=None):
    p=Path(path)
    if p.suffix=='.pt': return torch.load(p,weights_only=False)
    z=np.load(p); assert holdout_rank is not None and expert_map_fixture
    em=torch.load(expert_map_fixture,weights_only=False)['expert_map'].clone()
    xb=np.ascontiguousarray(z['x_bf16_bits']); x=torch.from_numpy(xb.view(np.int16)).view(torch.bfloat16)
    return {'schema':'ds41-holdout-npz-v1','rank':int(holdout_rank),'chunk_index':-1,'layer_index':0,'tokens':int(x.shape[0]),'x':x,'topk_ids':torch.from_numpy(np.ascontiguousarray(z['topk_ids'])).to(torch.int32),'topk_weights':torch.from_numpy(np.ascontiguousarray(z['topk_weights'])).float(),'expert_map':em,'swiglu_limit':10.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--lib',required=True); ap.add_argument('--out',required=True); ap.add_argument('--holdout-rank',type=int); ap.add_argument('--expert-map-fixture'); ap.add_argument('--sample-token',type=int,default=-1); a=ap.parse_args()
    fix=load_fix(a.input,a.holdout_rank,a.expert_map_fixture); T=int(fix['x'].shape[0]); H=int(fix['x'].shape[1]); MID=2304; li=int(fix['layer_index']); rank=int(fix['rank']); clamp=float(fix['swiglu_limit']); GEXP=384
    by=readers(); gt,ut,dt=[by[f'blk.{li}.{n}'] for n in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps')]
    emcpu=fix['expert_map'].to(torch.int64); local_by=sorted((int(l),g) for g,l in enumerate(emcpu.tolist()) if int(l)>=0); globals_idx=np.asarray([g for _,g in local_by],dtype=np.int64); LEXP=len(globals_idx)
    gate=torch.from_numpy(np.ascontiguousarray(gt.data[globals_idx])).cuda(); up=torch.from_numpy(np.ascontiguousarray(ut.data[globals_idx])).cuda(); down=torch.from_numpy(np.ascontiguousarray(dt.data[globals_idx])).cuda()
    x=fix['x'].cuda().contiguous(); gids=fix['topk_ids'].cuda().to(torch.int32).contiguous(); rw=fix['topk_weights'].cuda().float().contiguous(); em=fix['expert_map'].cuda().to(torch.int32).contiguous()
    lib=ctypes.CDLL(a.lib,mode=ctypes.RTLD_GLOBAL); lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int; assert lib.ds4_mmq_init(0)==0
    fn=lib.ds41_ds4_moe_full; fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*5+[ctypes.c_float]+[ctypes.c_void_p]*8; fn.restype=ctypes.c_int
    local_ids=torch.empty((T,6),device='cuda',dtype=torch.int32); xf=torch.empty((T,H),device='cuda',dtype=torch.float32); go=torch.empty((T*6,MID),device='cuda'); uo=torch.empty_like(go); mid=torch.empty_like(go); routes=torch.empty((T*6,H),device='cuda'); outf=torch.empty((T,H),device='cuda'); stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc=fn(ptr(x),ptr(gids),ptr(rw),ptr(em),ptr(gate),ptr(up),ptr(down),T,H,MID,GEXP,LEXP,ctypes.c_float(clamp),ptr(local_ids),ptr(xf),ptr(go),ptr(uo),ptr(mid),ptr(routes),ptr(outf),stream); assert rc==0,rc; torch.cuda.synchronize()
    # candidate final BF16 is the exact delivery boundary used in component benchmark
    cand=outf.to(torch.bfloat16).cpu().float().numpy(); ids=fix['topk_ids'].numpy(); weights=fix['topk_weights'].numpy(); xc=fix['x'].float().numpy(); local=fix['expert_map'][fix['topk_ids'].to(torch.long)].numpy()
    goh, uoh, mih, drh = go.cpu().numpy(), uo.cpu().numpy(), mid.cpu().numpy(), routes.cpu().numpy()
    # Deterministic sample token: requested, else last token having a local route.
    candidates=np.nonzero((local>=0).any(axis=1))[0]; assert len(candidates); ti=int(a.sample_token if a.sample_token>=0 else candidates[-1]); assert (local[ti]>=0).any()
    xhi=xc[ti].astype(np.float32); xq=q8_emulate(xhi,32)
    high=np.zeros(H,np.float32); emu=np.zeros(H,np.float32); stage=[]
    for slot,gid in enumerate(ids[ti].tolist()):
        lid=int(local[ti,slot]);
        if lid<0: continue
        wg=dequantize(np.ascontiguousarray(gt.data[gid]),gt.tensor_type).astype(np.float32,copy=False); wu=dequantize(np.ascontiguousarray(ut.data[gid]),ut.tensor_type).astype(np.float32,copy=False); wd=dequantize(np.ascontiguousarray(dt.data[gid]),dt.tensor_type).astype(np.float32,copy=False)
        g_hi=xhi@wg.T; u_hi=xhi@wu.T; m_hi=silu_weight(g_hi,u_hi,weights[ti,slot],clamp); y_hi=m_hi@wd.T; high+=y_hi.astype(np.float32)
        g_q=xq@wg.T; u_q=xq@wu.T; m_q=silu_weight(g_q,u_q,weights[ti,slot],clamp); mqq=q8_emulate(m_q.astype(np.float32),64); y_q=mqq@wd.T; emu+=y_q.astype(np.float32)
        ai=ti*6+slot
        self_mid=silu_weight(goh[ai],uoh[ai],weights[ti,slot],clamp)
        stage.append({'slot':slot,'gid':gid,'lid':lid,'gate_runtime_vs_q8ref':metric(goh[ai],g_q),'up_runtime_vs_q8ref':metric(uoh[ai],u_q),'mid_runtime_vs_runtime_epilogue':metric(mih[ai],self_mid),'mid_runtime_vs_q8ref':metric(mih[ai],m_q),'down_runtime_vs_q8ref':metric(drh[ai],y_q)})
    final_runtime=cand[ti]; final_emu=emu.astype(np.float32); final_high=high.astype(np.float32)
    # Row-level rel-L2 over every token against an inexpensive self reference is not independent; holdout/stress uses selected exact token. Save selected metrics.
    stvals=[max(s['gate_runtime_vs_q8ref']['rel_l2'],s['up_runtime_vs_q8ref']['rel_l2'],s['down_runtime_vs_q8ref']['rel_l2']) for s in stage]
    selfvals=[s['mid_runtime_vs_runtime_epilogue']['rel_l2'] for s in stage]
    m_emu=metric(final_runtime,final_emu); m_high=metric(final_runtime,final_high)
    map_exact=bool(torch.equal(local_ids.cpu(),fix['expert_map'][fix['topk_ids'].to(torch.long)]))
    passed=map_exact and np.isfinite(final_runtime).all() and max(stvals,default=0)<=G_STAGE and max(selfvals,default=0)<=G_SELF and m_emu['rel_l2']<=G_STAGE and m_high['rel_l2']<=G_FINAL
    out={'schema':'ds41-ds4-mmq-numeric-v1','status':'PASS' if passed else 'FAIL','input':a.input,'rank':rank,'layer':li,'tokens':T,'sample_token':ti,'ep2_map_exact':map_exact,'gates':{'stage_rel_l2_max':G_STAGE,'high_precision_final_rel_l2_max':G_FINAL,'epilogue_self_rel_l2_max':G_SELF},'q8_emulation':{'input_block':32,'down_mid_block':64,'round':'away_from_zero'},'stages':stage,'final_runtime_vs_bridge_emulation':m_emu,'final_runtime_vs_high_precision':m_high,'high_vs_bridge_emulation':metric(final_emu,final_high)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
