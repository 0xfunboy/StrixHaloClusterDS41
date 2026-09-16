#!/usr/bin/env python3
"""BLOCK_M4/8 gate on one captured real DS41 routed-MoE prefill fixture."""
from __future__ import annotations
import argparse, gc, json, math, os, statistics, time
from pathlib import Path
from types import SimpleNamespace
import gguf, numpy as np, torch
from gguf.quants import dequantize
from _ds41_artifact import MODEL_DIR
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

REL_CAND_BASE_MAX=2e-3; ABS_CAND_BASE_MAX=0.125
REL_INDEP_MAX=8e-2; ABS_INDEP_MAX=2.0; INDEP_REGRESSION_FACTOR=1.10
REPEATS=3
readers=[gguf.GGUFReader(str(p)) for p in sorted(Path(MODEL_DIR).glob('*.gguf'))]
by_name={t.name:t for r in readers for t in r.tensors}

def metric(a,b):
 d=a.float()-b.float(); rn=float(b.float().norm()); return {'max_abs':float(d.abs().max()),'rel_l2':float(d.norm())/max(rn,1e-30),'mean_abs':float(d.abs().mean())}
def layer(w13,w2,em): return SimpleNamespace(apply_router_weight_on_input=False,w13_weight=w13,w2_weight=w2,w13_weight_type=SimpleNamespace(weight_type=16),w2_weight_type=SimpleNamespace(weight_type=10),activation=SimpleNamespace(value='silu'),expert_map=em)
def method(limit): return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=limit))
def event_ms(fn):
 s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record(); y=fn(); e.record(); e.synchronize(); return y,float(s.elapsed_time(e))
def load_weights(layer_idx, em):
 globals_by_local=sorted(((int(local),g) for g,local in enumerate(em.tolist()) if int(local)>=0))
 assert [x[0] for x in globals_by_local]==list(range(len(globals_by_local)))
 idx=np.asarray([g for _,g in globals_by_local],dtype=np.int64)
 gt=by_name[f'blk.{layer_idx}.ffn_gate_exps']; ut=by_name[f'blk.{layer_idx}.ffn_up_exps']; dt=by_name[f'blk.{layer_idx}.ffn_down_exps']
 gate=torch.from_numpy(np.ascontiguousarray(gt.data[idx])).to('cuda'); up=torch.from_numpy(np.ascontiguousarray(ut.data[idx])).to('cuda'); down=torch.from_numpy(np.ascontiguousarray(dt.data[idx])).to('cuda'); w13=torch.cat((gate,up),dim=1); del gate,up
 return w13,down,(gt,ut,dt)
def indep_rows(fix,tensors,token_indices):
 gt,ut,dt=tensors; x=fix['x'].to('cuda'); ids=fix['topk_ids'].to(torch.int64); wt=fix['topk_weights'].to('cuda'); em=fix['expert_map'].to(torch.int64); limit=float(fix['swiglu_limit']); refs={}
 for ti in token_indices:
  acc=torch.zeros((x.shape[1],),device='cuda',dtype=torch.float32)
  for slot,gid in enumerate(ids[ti].tolist()):
   if gid<0 or gid>=em.numel() or int(em[gid])<0: continue
   wg=torch.from_numpy(dequantize(np.ascontiguousarray(gt.data[gid]),gt.tensor_type).astype(np.float32,copy=False)).to('cuda')
   wu=torch.from_numpy(dequantize(np.ascontiguousarray(ut.data[gid]),ut.tensor_type).astype(np.float32,copy=False)).to('cuda')
   wd=torch.from_numpy(dequantize(np.ascontiguousarray(dt.data[gid]),dt.tensor_type).astype(np.float32,copy=False)).to('cuda')
   xv=x[ti:ti+1].float(); ga=(xv@wg.T).to(torch.bfloat16).float(); up=(xv@wu.T).to(torch.bfloat16).float(); ga=torch.clamp(ga,max=limit); up=torch.clamp(up,-limit,limit); act=(ga*torch.sigmoid(ga)*up).to(torch.bfloat16); y=(act.float()@wd.T).to(torch.bfloat16).float()[0]; acc.add_(y*wt[ti,slot].float()); del wg,wu,wd,ga,up,act,y
  refs[int(ti)]=acc.to(torch.bfloat16).cpu()
 return refs
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fixture',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); fix=torch.load(a.fixture,weights_only=False); em=fix['expert_map'].to(torch.int64); w13,w2,tensors=load_weights(int(fix['layer_index']),em); x=fix['x'].to('cuda'); weights=fix['topk_weights'].to('cuda'); ids=fix['topk_ids'].to('cuda'); L=layer(w13,w2,em.to('cuda')); M=method(float(fix['swiglu_limit']))
 outputs={}; timings={}; mem={}
 for bm in (4,8):
  os.environ['DS41_MOE_PREFILL_BLOCK_M']=str(bm); GGUFMoEMethod.apply(M,L,x,weights,ids,None,None); torch.cuda.synchronize(); vals=[]; last=None; torch.cuda.reset_peak_memory_stats(); base=torch.cuda.memory_allocated()
  for _ in range(REPEATS): last,ms=event_ms(lambda:GGUFMoEMethod.apply(M,L,x,weights,ids,None,None)); vals.append(ms)
  outputs[bm]=last.detach().cpu(); timings[bm]={'samples_ms':vals,'median_ms':statistics.median(vals),'mean_ms':statistics.mean(vals)}; mem[bm]={'base_allocated':base,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_delta':torch.cuda.max_memory_allocated()-base}
 cb=metric(outputs[8],outputs[4])
 candidate_baseline_pass = cb['rel_l2']<=REL_CAND_BASE_MAX and cb['max_abs']<=ABS_CAND_BASE_MAX
 # Two deterministic real tokens carrying at least one local route.
 localmask=em[fix['topk_ids'].to(torch.int64)]>=0; candidates=torch.nonzero(localmask.any(dim=1),as_tuple=False).flatten().tolist(); assert candidates; sample=[candidates[0],candidates[-1]] if len(candidates)>1 else [candidates[0]]; refs=indep_rows(fix,tensors,sample); indep={}
 for bm in (4,8):
  ms=[]
  for ti,ref in refs.items(): ms.append(metric(outputs[bm][ti],ref))
  indep[bm]={'rows':ms,'max_rel_l2':max(m['rel_l2'] for m in ms),'max_abs':max(m['max_abs'] for m in ms)}
 independent_pass = all(indep[bm]['max_rel_l2']<=REL_INDEP_MAX and indep[bm]['max_abs']<=ABS_INDEP_MAX for bm in (4,8))
 independent_regression_pass = indep[8]['max_rel_l2'] <= indep[4]['max_rel_l2']*INDEP_REGRESSION_FACTOR + 1e-7
 numeric_pass = candidate_baseline_pass and independent_pass and independent_regression_pass
 gain=(timings[4]['median_ms']/timings[8]['median_ms']-1)*100
 out={'schema':'ds41-blockm-real-fixture-v1','status':'PASS' if numeric_pass else 'FAIL_NUMERIC','fixture':a.fixture,'rank':int(fix['rank']),'chunk_index':int(fix['chunk_index']),'layer_index':int(fix['layer_index']),'tokens':int(fix['tokens']),'gates':{'candidate_baseline_rel_l2_max':REL_CAND_BASE_MAX,'candidate_baseline_max_abs':ABS_CAND_BASE_MAX,'independent_rel_l2_max':REL_INDEP_MAX,'independent_max_abs':ABS_INDEP_MAX,'independent_regression_factor':INDEP_REGRESSION_FACTOR},'candidate_vs_baseline':cb,'numeric_gate':{'candidate_baseline_pass':candidate_baseline_pass,'independent_pass':independent_pass,'independent_regression_pass':independent_regression_pass,'pass':numeric_pass},'independent':indep,'timing':timings,'gain_pct_by_median':gain,'memory':mem,'sampled_reference_tokens':sample}
 p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
