#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes
from pathlib import Path
import numpy as np, torch, gguf
from _ds41_artifact import MODEL_DIR

def ptr(t): return ctypes.c_void_p(t.data_ptr())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fixture',required=True); ap.add_argument('--lib',required=True); a=ap.parse_args()
 f=torch.load(a.fixture,weights_only=False); T=int(f['tokens']); H=f['x'].shape[1]; M=2304; G=384; emc=f['expert_map'].long(); local=sorted((int(l),g) for g,l in enumerate(emc.tolist()) if int(l)>=0); idx=np.asarray([g for _,g in local],np.int64); L=len(idx)
 rs=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]; by={t.name:t for r in rs for t in r.tensors}; li=int(f['layer_index']); gt,ut,dt=[by[f'blk.{li}.{n}'] for n in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps')]
 gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).cuda(); up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).cuda(); w13=torch.cat((gate,up),dim=1).contiguous(); down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).cuda(); x=f['x'].cuda().contiguous(); ids=f['topk_ids'].cuda().int().contiguous(); rw=f['topk_weights'].cuda().float().contiguous(); em=f['expert_map'].cuda().int().contiguous(); clamp=float(f['swiglu_limit'])
 lib=ctypes.CDLL(a.lib,mode=ctypes.RTLD_GLOBAL); lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int; assert lib.ds4_mmq_init(0)==0
 P=ctypes.c_void_p; common=[P,P,P,P]
 f0=lib.ds41_ds4_moe_full; f0.argtypes=[P]*7+[ctypes.c_int]*5+[ctypes.c_float]+[P]*8; f0.restype=ctypes.c_int
 f1=lib.ds41_ds4_moe_full_w13; f1.argtypes=[P]*6+[ctypes.c_int]*5+[ctypes.c_float]+[P]*7; f1.restype=ctypes.c_int
 def scratch(): return (torch.empty((T,6),device='cuda',dtype=torch.int32),torch.empty((T,H),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,M),device='cuda'),torch.empty((T*6,H),device='cuda'),torch.empty((T,H),device='cuda'))
 s0=scratch(); s1=scratch(); stream=P(torch.cuda.current_stream().cuda_stream)
 r=f0(ptr(x),ptr(ids),ptr(rw),ptr(em),ptr(gate),ptr(up),ptr(down),T,H,M,G,L,ctypes.c_float(clamp),*[ptr(t) for t in s0],stream); assert r==0,r
 r=f1(ptr(x),ptr(ids),ptr(rw),ptr(em),ptr(w13),ptr(down),T,H,M,G,L,ctypes.c_float(clamp),*[ptr(t) for t in s1],stream); assert r==0,r
 torch.cuda.synchronize()
 names=['local_ids','x_f32','gate','up','mid','down_routes','out_f32']; diff={n:{'exact':bool(torch.equal(a,b)),'max_abs':float((a.float()-b.float()).abs().max())} for n,a,b in zip(names,s0,s1)}
 assert all(v['exact'] for v in diff.values()),diff
 print({'status':'PASS','fixture':a.fixture,'diff':diff})
if __name__=='__main__': main()
