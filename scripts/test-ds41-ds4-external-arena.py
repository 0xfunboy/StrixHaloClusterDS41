#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes,json
from pathlib import Path
import numpy as np, torch, gguf
from _ds41_artifact import MODEL_DIR

def ptr(t): return ctypes.c_void_p(t.data_ptr())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fixture',required=True); ap.add_argument('--lib',required=True); ap.add_argument('--out-tensor'); ap.add_argument('--arena-mib',type=int,default=64); ap.add_argument('--repeats',type=int,default=3); a=ap.parse_args()
 assert 1 <= a.repeats <= 16
 f=torch.load(a.fixture,weights_only=False); T=int(f['tokens']); H=int(f['x'].shape[1]); M=2304; G=384
 emc=f['expert_map'].long(); local=sorted((int(l),g) for g,l in enumerate(emc.tolist()) if int(l)>=0); idx=np.asarray([g for _,g in local],np.int64); L=len(idx)
 rs=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]; by={t.name:t for r in rs for t in r.tensors}; li=int(f['layer_index']); gt,ut,dt=[by[f'blk.{li}.{n}'] for n in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps')]
 gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).cuda(); up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).cuda(); w13=torch.cat((gate,up),dim=1).contiguous(); down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).cuda(); x=f['x'].cuda().contiguous(); ids=f['topk_ids'].cuda().int().contiguous(); rw=f['topk_weights'].cuda().float().contiguous(); em=f['expert_map'].cuda().int().contiguous(); clamp=float(f['swiglu_limit'])
 lib=ctypes.CDLL(a.lib,mode=ctypes.RTLD_LOCAL); P=ctypes.c_void_p
 arena=torch.empty((a.arena_mib*1024*1024,),device='cuda',dtype=torch.uint8)
 lib.ds4_pool_set_external_arena.argtypes=[P,ctypes.c_size_t]; lib.ds4_pool_set_external_arena.restype=ctypes.c_int
 lib.ds4_pool_external_arena_high_water.restype=ctypes.c_size_t; lib.ds4_pool_external_arena_allocs.restype=ctypes.c_uint64; lib.ds4_pool_external_arena_in_use.restype=ctypes.c_size_t
 assert lib.ds4_pool_set_external_arena(ptr(arena),arena.numel())==0
 lib.ds4_mmq_init.argtypes=[ctypes.c_int]; lib.ds4_mmq_init.restype=ctypes.c_int; assert lib.ds4_mmq_init(0)==0
 fn=lib.ds41_ds4_moe_full_w13; fn.argtypes=[P]*6+[ctypes.c_int]*5+[ctypes.c_float]+[P]*8; fn.restype=ctypes.c_int
 local_ids=torch.empty((T,6),device='cuda',dtype=torch.int32); xf=torch.empty((T,H),device='cuda'); gateo=torch.empty((T*6,M),device='cuda'); upo=torch.empty_like(gateo); mid=torch.empty_like(gateo); routes=torch.empty((T*6,H),device='cuda'); out=torch.empty((T,H),device='cuda')
 stream=P(torch.cuda.current_stream().cuda_stream)
 rc=0; repeat_exact=True; first_out=None
 for _ in range(a.repeats):
  rc=fn(ptr(x),ptr(ids),ptr(rw),ptr(em),ptr(w13),ptr(down),T,H,M,G,L,ctypes.c_float(clamp),ptr(local_ids),ptr(xf),ptr(gateo),ptr(upo),ptr(mid),ptr(routes),ptr(out),stream)
  if rc: break
  torch.cuda.synchronize()
  if first_out is None: first_out=out.clone()
  else: repeat_exact=repeat_exact and bool(torch.equal(out,first_out))
 high=int(lib.ds4_pool_external_arena_high_water()); allocs=int(lib.ds4_pool_external_arena_allocs()); in_use=int(lib.ds4_pool_external_arena_in_use())
 finite=bool(torch.isfinite(out).all()); map_exact=bool(torch.equal(local_ids,em[ids.long()]))
 if a.out_tensor: torch.save(out.to(torch.bfloat16).cpu(),a.out_tensor)
 res={'status':'PASS' if rc==0 and finite and map_exact and repeat_exact and in_use==0 and 0<high<=arena.numel() else 'FAIL','rc':rc,'tokens':T,'repeats':a.repeats,'repeat_exact':repeat_exact,'arena_bytes':arena.numel(),'high_water_bytes':high,'allocs':allocs,'in_use_bytes':in_use,'finite':finite,'ep2_map_exact':map_exact}
 print(json.dumps(res,sort_keys=True)); assert res['status']=='PASS',res
if __name__=='__main__': main()
