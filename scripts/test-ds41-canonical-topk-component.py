#!/usr/bin/env python3
from __future__ import annotations
import json,statistics
import torch

def canonical(x):
    sentinel=torch.iinfo(x.dtype).max
    y=torch.where(x>=0,x,torch.full_like(x,sentinel))
    y=torch.sort(y,dim=-1,stable=True).values
    return torch.where(y==sentinel,torch.full_like(y,-1),y)
def event(fn):
 s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record(); y=fn(); e.record(); e.synchronize(); return y,float(s.elapsed_time(e))

def test_shape(n):
 torch.manual_seed(1)
 # Realistic valid-width pattern: rows grow to <=512, invalid suffix is -1.
 x=torch.full((n,512),-1,dtype=torch.int32,device='cuda')
 for i in range(n):
  k=min(512,max(1,(i+1)//2)); vals=torch.randperm(max(k+64,512),device='cuda',dtype=torch.int64)[:k].to(torch.int32); x[i,:k]=vals
 # scramble valid prefix so canonicalizer has work
 for i in range(n):
  k=int((x[i]>=0).sum()); x[i,:k]=x[i,:k][torch.randperm(k,device='cuda')]
 y=canonical(x); torch.cuda.synchronize()
 assert torch.equal((x>=0).sum(1),(y>=0).sum(1))
 # Sorted valid values preserve set/multiset exactly.
 for i in (0,n//2,n-1):
  a=x[i][x[i]>=0].sort().values; b=y[i][y[i]>=0]; assert torch.equal(a,b)
 vals=[]
 for _ in range(2): event(lambda:canonical(x))
 for _ in range(7): _,ms=event(lambda:canonical(x)); vals.append(ms)
 return {'rows':n,'width':512,'median_ms':statistics.median(vals),'mean_ms':statistics.mean(vals),'samples_ms':vals,'valid_counts_minmax':[int((x>=0).sum(1).min()),int((x>=0).sum(1).max())]}
out={'schema':'ds41-canonical-topk-component-v1','status':'PASS','shapes':[test_shape(1023),test_shape(565)]}
print(json.dumps(out,indent=2))
