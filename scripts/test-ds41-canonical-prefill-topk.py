#!/usr/bin/env python3
from types import SimpleNamespace
import os
import torch
from vllm.models.deepseek_v4_1.attention import _ds41_canonicalize_prefill_topk, _ds41_maybe_canonicalize_prefill_topk

def check(x):
    before=x.clone(); _ds41_canonicalize_prefill_topk(x)
    assert torch.equal((before>=0).sum(-1),(x>=0).sum(-1))
    for i in range(x.shape[0]):
        assert torch.equal(before[i][before[i]>=0].sort().values,x[i][x[i]>=0])
        if (x[i]<0).any(): assert (x[i][x[i]<0]==-1).all()
    return x
x=torch.tensor([[5,2,9,-1,-1],[3,1,2,0,-1]],device='cuda',dtype=torch.int32)
y=check(x); assert y.tolist()==[[2,5,9,-1,-1],[0,1,2,3,-1]]
# Empty and already canonical paths.
check(torch.empty((0,512),device='cuda',dtype=torch.int32))
z=torch.tensor([[0,4,8,-1]],device='cuda',dtype=torch.int32); old=z.clone(); check(z); assert torch.equal(z,old)

# Mixed decode+prefill guard: only the real prefill slice is canonicalized.
buf=torch.tensor([[9,1,4,-1,-1],[7,3,6,-1,-1],[8,2,5,-1,-1],[11,0,4,-1,-1],[6,5,1,-1,-1],[99,98,97,-1,-1]],device='cuda',dtype=torch.int32)
orig=buf.clone(); meta=SimpleNamespace(num_decode_tokens=2,num_prefill_tokens=3)
os.environ['DS41_CANONICAL_PREFILL_TOPK']='0'; assert _ds41_maybe_canonicalize_prefill_topk(buf,meta,5) is None; assert torch.equal(buf,orig)
os.environ['DS41_CANONICAL_PREFILL_TOPK']='1'; assert _ds41_maybe_canonicalize_prefill_topk(buf,meta,5)==(2,3)
assert torch.equal(buf[:2],orig[:2]); assert torch.equal(buf[5:],orig[5:])
assert buf[2:5].tolist()==[[2,5,8,-1,-1],[0,4,11,-1,-1],[1,5,6,-1,-1]]
z=orig.clone(); assert _ds41_maybe_canonicalize_prefill_topk(z,SimpleNamespace(num_decode_tokens=2,num_prefill_tokens=0),5) is None; assert torch.equal(z,orig)

print('DS41_CANONICAL_PREFILL_TOPK_MODEL_FREE=PASS')
