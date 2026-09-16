#!/usr/bin/env python3
import os, torch
import runtime.ds41.ds4_mmq_runtime as r
class Q: 
    def __init__(self,v): self.weight_type=v
class A: value='silu'
class L: pass
L0=L(); L0.expert_map=torch.arange(384,dtype=torch.int32); L0.w13_weight=torch.empty((1,4608,1),dtype=torch.uint8); L0.w2_weight=torch.empty((1,5120,1),dtype=torch.uint8); L0.w13_weight_type=Q(16); L0.w2_weight_type=Q(10); L0.activation=A()
def check(pair,T,expect,reason=None):
    old=r._metadata_counts; r._metadata_counts=lambda:pair
    x=torch.empty((T,5120),dtype=torch.bfloat16); ids=torch.zeros((T,6),dtype=torch.int32); w=torch.ones((T,6),dtype=torch.float32)
    ok,why=r._eligible(L0,x,w,ids,16,10,'silu')
    r._metadata_counts=old
    assert ok==expect,(pair,T,ok,why)
    if reason: assert why==reason,(why,reason)
os.environ['DS41_DS4_MMQ_PREFILL']='1'; os.environ['DS41_DS4_MMQ_MIN_TOKENS']='128'; os.environ['DS41_DS4_MMQ_MAX_TOKENS']='1024'
check((0,128),128,True)
check((0,1024),1024,True)
check((1,127),128,False,'not_pure_target_prefill')
check((4,0),4,False,'not_pure_target_prefill')
check((0,4),4,False,'prefill_tokens_outside_admitted_range')
check((0,1025),1025,False,'prefill_tokens_outside_admitted_range')
os.environ['DS41_DS4_MMQ_PREFILL']='0'; old=r._metadata_counts; r._metadata_counts=lambda:(0,128); x=torch.empty((128,5120),dtype=torch.bfloat16); ids=torch.zeros((128,6),dtype=torch.int32); w=torch.ones((128,6)); ok,why=r._eligible(L0,x,w,ids,16,10,'silu'); r._metadata_counts=old; assert not ok and why=='disabled'
print('DS41_DS4_DISPATCH_MODEL_FREE=PASS')
