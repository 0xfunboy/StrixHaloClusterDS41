#!/usr/bin/env python3
"""Model-free DS41 main-attention WQ_B M=1 skinny-GEMM gate on real layer0 Q input."""
from __future__ import annotations
import importlib.util, json, os, time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from vllm.utils.platform_utils import num_compute_units

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=Path(os.environ.get('DS41_WQB_SKINNY_OUT',str(ROOT/'reports/DS41-Q2-001/perf/attention-wqb-skinny-node01.json')))
REPEATS=int(os.environ.get('DS41_WQB_REPEATS','300')); REL_MAX=5e-3; ABS_MAX=0.125

def l0():
 p=ROOT/'scripts/test-ds41-layer0-complete-densefix.py'; s=importlib.util.spec_from_file_location('l0',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def metric(a,b):
 af,bf=a.float(),b.float(); d=af-bf; rn=float(bf.norm()); return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'rel_l2':float(d.norm())/max(rn,1e-30),'finite':bool(torch.isfinite(af).all() and torch.isfinite(bf).all())}

def timed(fn):
 for _ in range(20): fn()
 torch.cuda.synchronize(); ev=[]; t0=time.perf_counter_ns()
 for _ in range(REPEATS):
  a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); ev.append((a,b))
 torch.cuda.synchronize(); t1=time.perf_counter_ns(); vals=[a.elapsed_time(b) for a,b in ev]
 return {'repeats':REPEATS,'gpu_ms_mean':sum(vals)/len(vals),'gpu_ms_min':min(vals),'gpu_ms_max':max(vals),'wall_ms_mean':(t1-t0)/1e6/REPEATS}

def build_qrn(m):
 os.environ['DS41_MHC_COEFF_SINKHORN']='1'; os.environ['DS41_MHC_PROJECTION_RMS']='1'
 ids=torch.tensor(json.loads(m.PROMPT.read_text())['prompts']['arithmetic']['token_ids'],dtype=torch.long,device='cuda'); linear=m.UnquantizedLinearMethod(); emb=m.bf16('token_embd')[ids]
 afn,asc,abase=m.f32('blk.0.hc_attn_fn'),m.f32('blk.0.hc_attn_scale'),m.f32('blk.0.hc_attn_base'); residual=emb[:,None,:].expand(-1,m.HC,-1).contiguous(); afn_b=afn.reshape(24,m.HC,m.H).sum(1)
 _,_,x,_=m.mhc_pre_delayed_torch(residual,afn_b,asc,abase,m.EPS,m.HC_EPS,m.HC_EPS,2.0,m.SINK_ITERS,x=emb); xn=m.rms_candidate(x,m.bf16('blk.0.attn_norm'))
 fused=torch.cat((m.bf16('blk.0.attn_q_a'),m.bf16('blk.0.attn_kv')),dim=0).contiguous(); qkv=linear.apply(SimpleNamespace(weight=fused),xn); qra,_=qkv.split([m.QRA,m.HD],dim=-1)
 return m.rms_candidate(qra,m.bf16('blk.0.attn_q_a_norm'))

def main():
 assert torch.cuda.is_available(); m=l0(); xall=build_qrn(m); wfull=m.bf16('blk.0.attn_q_b').contiguous(); assert tuple(wfull.shape)==(m.HEADS*m.HD,m.QRA)
 cu=num_compute_units(); rows=[]
 for rank in (0,1):
  lo=rank*(wfull.shape[0]//2); hi=(rank+1)*(wfull.shape[0]//2); w=wfull[lo:hi].contiguous(); mets=[]
  qualified={'LLMM1':True,'wvSplitK':True}
  for ti in range(xall.shape[0]-4,xall.shape[0]):
   x=xall[ti:ti+1].contiguous(); ref=F.linear(x,w); ll=m.vllm_ops.LLMM1(w,x,4); sp=m.vllm_ops.wvSplitK(w,x,cu,None)
   ml,ms=metric(ll,ref),metric(sp,ref)
   qualified['LLMM1'] &= ml['finite'] and ml['rel_l2']<=REL_MAX and ml['max_abs']<=ABS_MAX
   qualified['wvSplitK'] &= ms['finite'] and ms['rel_l2']<=REL_MAX and ms['max_abs']<=ABS_MAX
   mets.append({'token_index':ti,'llmm1':ml,'wvsplitk':ms})
  x=xall[-1:].contiguous(); bt=timed(lambda:F.linear(x,w)); lt=timed(lambda:m.vllm_ops.LLMM1(w,x,4)); st=timed(lambda:m.vllm_ops.wvSplitK(w,x,cu,None))
  candidates=[]
  if qualified['LLMM1']: candidates.append(('LLMM1',lt))
  if qualified['wvSplitK']: candidates.append(('wvSplitK',st))
  if not candidates: raise RuntimeError(f'rank{rank}: no skinny primitive passed fixed gates: {qualified}')
  chosen,ct=min(candidates,key=lambda item:item[1]['gpu_ms_mean'])
  rows.append({'rank':rank,'cu_count':cu,'x_shape':list(x.shape),'weight_shape':list(w.shape),'metrics':mets,'qualified':qualified,'baseline':bt,'llmm1':lt,'wvsplitk':st,'chosen':chosen,'candidate':ct,'gpu_speedup':bt['gpu_ms_mean']/ct['gpu_ms_mean'],'wall_speedup':bt['wall_ms_mean']/ct['wall_ms_mean']})
 out={'status':'PASS','gate':{'rel_l2_max':REL_MAX,'max_abs':ABS_MAX},'rows':rows}; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
