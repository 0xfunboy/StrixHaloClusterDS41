#!/usr/bin/env python3
from __future__ import annotations
import argparse, ctypes, json
from pathlib import Path
import numpy as np, torch, gguf
from gguf.quants import dequantize
from _ds41_artifact import MODEL_DIR
G_ROW95=.08; G_FINAL=.05

def ptr(t): return ctypes.c_void_p(t.data_ptr())
def metric_rows(a,b):
 a=a.float(); b=b.float(); d=a-b; den=b.norm(dim=1).clamp_min(1e-30); rel=(d.norm(dim=1)/den).cpu().numpy(); ma=d.abs().amax(dim=1).cpu().numpy(); scale=b.abs().amax(dim=1).clamp_min(1e-30).cpu().numpy();
 return {'rel_l2':rel.tolist(),'rel_l2_p95':float(np.percentile(rel,95)),'rel_l2_max':float(rel.max()),'max_abs_max':float(ma.max()),'max_abs_over_scale_p95':float(np.percentile(ma/scale,95))}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--lib',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); rank=a.rank
 z=np.load('reports/DS41-Q2-001/perf/native-hip/layer0-moe-input.npz'); xb=np.ascontiguousarray(z['x_bf16_bits']); x=torch.from_numpy(xb.view(np.int16)).view(torch.bfloat16); ids=torch.from_numpy(np.ascontiguousarray(z['topk_ids'])).to(torch.int32); rw=torch.from_numpy(np.ascontiguousarray(z['topk_weights'])).float(); T,H=x.shape; M=2304; G=384
 f=torch.load(f'reports/DS41-Q2-001/blockm-real-fixtures-bc41a35/route-fixture-rank{rank}-call00.pt',weights_only=False); emc=f['expert_map'].to(torch.int64); local_by=sorted((int(l),g) for g,l in enumerate(emc.tolist()) if int(l)>=0); idx=np.asarray([g for _,g in local_by],np.int64); L=len(idx)
 rs=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]; by={t.name:t for r in rs for t in r.tensors}; gt,ut,dt=[by[f'blk.0.{n}'] for n in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps')]
 gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).cuda(); up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).cuda(); down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).cuda(); xd=x.cuda().contiguous(); idd=ids.cuda().contiguous(); rwd=rw.cuda().contiguous(); em=f['expert_map'].cuda().int().contiguous()
 lib=ctypes.CDLL(a.lib,mode=ctypes.RTLD_GLOBAL); lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int; assert lib.ds4_mmq_init(0)==0; fn=lib.ds41_ds4_moe_full; P=ctypes.c_void_p; fn.argtypes=[P]*7+[ctypes.c_int]*5+[ctypes.c_float]+[P]*8; fn.restype=ctypes.c_int
 s=(torch.empty((T,6),device='cuda',dtype=torch.int32),torch.empty((T,H),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,H),device='cuda'),torch.empty((T,H),device='cuda')); stream=P(torch.cuda.current_stream().cuda_stream); rc=fn(ptr(xd),ptr(idd),ptr(rwd),ptr(em),ptr(gate),ptr(up),ptr(down),T,H,M,G,L,ctypes.c_float(10.0),*[ptr(q) for q in s],stream); assert rc==0,rc; torch.cuda.synchronize(); cand=s[-1].to(torch.bfloat16).clone()
 # Independent high-precision dequant reference, batched by expert; no Q8 activation quantization.
 ref=torch.zeros((T,H),device='cuda',dtype=torch.float32); ids_np=ids.numpy(); em_np=f['expert_map'].numpy(); xfp=xd.float()
 unique=sorted(set(int(g) for g in ids_np.reshape(-1) if 0<=int(g)<G and int(em_np[int(g)])>=0))
 for gid in unique:
  where=np.argwhere(ids_np==gid)
  if where.size==0: continue
  toks=torch.tensor(where[:,0],device='cuda',dtype=torch.long); slots=torch.tensor(where[:,1],device='cuda',dtype=torch.long)
  wg=torch.from_numpy(dequantize(np.ascontiguousarray(gt.data[gid]),gt.tensor_type).astype(np.float32,copy=False)).cuda(); wu=torch.from_numpy(dequantize(np.ascontiguousarray(ut.data[gid]),ut.tensor_type).astype(np.float32,copy=False)).cuda(); wd=torch.from_numpy(dequantize(np.ascontiguousarray(dt.data[gid]),dt.tensor_type).astype(np.float32,copy=False)).cuda()
  xx=xfp[toks]; gg=xx@wg.T; uu=xx@wu.T; gg=torch.clamp(gg,max=10.0); uu=torch.clamp(uu,-10.0,10.0); mid=(gg*torch.sigmoid(gg)*uu); yy=mid@wd.T; ww=rwd[toks,slots].view(-1,1); ref.index_add_(0,toks,yy*ww)
  del wg,wu,wd,xx,gg,uu,mid,yy; torch.cuda.empty_cache()
 met=metric_rows(cand.float(),ref); full=(cand.float()-ref).norm()/ref.norm().clamp_min(1e-30); passed=met['rel_l2_p95']<=G_ROW95 and float(full)<=G_FINAL and bool(torch.isfinite(cand).all()) and bool(torch.equal(s[0].cpu(),f['expert_map'][ids.long()]))
 out={'schema':'ds41-ds4-holdout-rows-v1','status':'PASS' if passed else 'FAIL','rank':rank,'tokens':int(T),'unique_local_experts':len(unique),'full_rel_l2':float(full),'per_row':met,'gates':{'full_rel_l2_max':G_FINAL,'row_rel_l2_p95_max':G_ROW95}}
 Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
