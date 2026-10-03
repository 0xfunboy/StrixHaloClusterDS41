#!/usr/bin/env python3
"""Model-free DS41 WO_A M=1 Triton grouped-GEMV gate on real layer0 data."""
from __future__ import annotations
import importlib.util, json, os, time
from pathlib import Path
import torch

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
OUT=Path(os.environ.get('DS41_WOA_TRITON_OUT',str(ROOT/'reports/DS41-Q2-001/perf/attention-woa-triton-node01.json')))
REL_MAX=5e-3; ABS_MAX=0.125; REPEATS=int(os.environ.get('DS41_WOA_TRITON_REPEATS','300'))

def load_helper():
 p=ROOT/'scripts/test-ds41-attn-woa-bmm.py'; s=importlib.util.spec_from_file_location('woa_bmm',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def metric(a,b):
 af,bf=a.float(),b.float(); d=af-bf; rn=float(bf.norm()); return {'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'rel_l2':float(d.norm())/max(rn,1e-30),'finite':bool(torch.isfinite(af).all() and torch.isfinite(bf).all())}

def timed(fn):
 for _ in range(20): fn()
 torch.cuda.synchronize(); ev=[]; t0=time.perf_counter_ns()
 for _ in range(REPEATS):
  a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); ev.append((a,b))
 torch.cuda.synchronize(); t1=time.perf_counter_ns(); vals=[a.elapsed_time(b) for a,b in ev]
 return {'repeats':REPEATS,'gpu_ms_mean':sum(vals)/len(vals),'gpu_ms_min':min(vals),'gpu_ms_max':max(vals),'wall_ms_mean':(t1-t0)/1e6/REPEATS}

def main():
 assert torch.cuda.is_available()
 h=load_helper(); l0=h.load_l0(); ao,pos,cache=h.build_attention_output(l0)
 from runtime.ds41.attn_woa_gemv import woa_m1_gemv
 woa=l0.bf16('blk.0.attn_output_a').reshape(l0.GROUPS,l0.ORANK,l0.HEADS*l0.HD//l0.GROUPS).contiguous()
 rows=[]
 for rank in (0,1):
  gs=slice(rank*4,(rank+1)*4); w=woa[gs].contiguous(); metrics=[]
  for ti in range(ao.shape[0]-4,ao.shape[0]):
   o=ao[ti:ti+1].contiguous(); p=pos[ti:ti+1]
   inv=l0._fused_inverse_rope_gptj(o,p,cache,l0.ROPE).view(1,l0.GROUPS,-1)[:,gs,:].contiguous()
   ref=torch.einsum('tgd,grd->tgr',inv,w); cand=woa_m1_gemv(inv,w); m=metric(cand,ref)
   if not m['finite'] or m['rel_l2']>REL_MAX or m['max_abs']>ABS_MAX: raise RuntimeError((rank,ti,m))
   metrics.append({'token_index':ti,**m})
  o=ao[-1:].contiguous(); p=pos[-1:]; inv=l0._fused_inverse_rope_gptj(o,p,cache,l0.ROPE).view(1,l0.GROUPS,-1)[:,gs,:].contiguous()
  def base(): return torch.einsum('tgd,grd->tgr',inv,w)
  def cand(): return woa_m1_gemv(inv,w)
  bt=timed(base); ct=timed(cand)
  rows.append({'rank':rank,'x_shape':list(inv.shape),'weight_shape':list(w.shape),'metrics':metrics,'baseline':bt,'candidate':ct,'gpu_speedup':bt['gpu_ms_mean']/ct['gpu_ms_mean'],'wall_speedup':bt['wall_ms_mean']/ct['wall_ms_mean']})
 out={'status':'PASS','gate':{'rel_l2_max':REL_MAX,'max_abs':ABS_MAX},'rows':rows}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
