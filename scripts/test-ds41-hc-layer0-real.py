#!/usr/bin/env python3
"""Compare DS41 layer0 mHC torch path with an independent NumPy/Vontra formula."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch, mhc_post_torch

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
MODEL=Path('/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2')
H=5120; HC=4; MIX=(2+HC)*HC; RMS_EPS=1e-20; HC_EPS=1e-6; ITERS=20; POST_ALPHA=2.0
TOKEN_IDS=[0,128803,128804,128822,66851]

def get_tensor(name):
    for p in sorted(MODEL.glob('*.gguf')):
        r=GGUFReader(str(p))
        for t in r.tensors:
            if t.name==name:
                a=t.data
                if t.tensor_type.name=='BF16':
                    a=a.view(np.uint16)
                    # Convert exact BF16 bits to float32 without depending on NumPy bfloat16.
                    u32=a.astype(np.uint32) << 16
                    return u32.view(np.float32).reshape(*(int(x) for x in reversed(t.shape))).copy()
                return np.asarray(a).reshape(*(int(x) for x in reversed(t.shape))).copy()
    raise KeyError(name)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def np_hc_mix(residual, fn, scale, base):
    f=residual.reshape(residual.shape[0],-1).astype(np.float32)
    mixes=(f @ fn.T) / np.sqrt(np.mean(f*f,axis=-1,keepdims=True)+RMS_EPS)
    pre=sigmoid(mixes[:,:HC]*scale[0]+base[:HC])+HC_EPS
    post=POST_ALPHA*sigmoid(mixes[:,HC:2*HC]*scale[1]+base[HC:2*HC])
    c=(mixes[:,2*HC:]*scale[2]+base[2*HC:]).reshape(-1,HC,HC)
    # stable softmax
    c=np.exp(c-c.max(axis=-1,keepdims=True)); c/=c.sum(axis=-1,keepdims=True)
    c=c+HC_EPS
    c=c/(c.sum(axis=-2,keepdims=True)+HC_EPS)
    for _ in range(ITERS-1):
        c=c/(c.sum(axis=-1,keepdims=True)+HC_EPS)
        c=c/(c.sum(axis=-2,keepdims=True)+HC_EPS)
    return pre.astype(np.float32),post.astype(np.float32),c.astype(np.float32)

def np_hc_post(y,residual,post,comb):
    return (post[...,None]*y[:,None,:].astype(np.float32)+np.einsum('bij,bid->bjd',comb,residual.astype(np.float32))).astype(np.float32)

def metrics(a,b):
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32); d=np.abs(a-b)
    denom=np.maximum(np.abs(b),1e-12)
    return {'max_abs':float(d.max(initial=0)), 'mean_abs':float(d.mean()), 'max_rel':float((d/denom).max(initial=0))}

def main():
    emb=get_tensor('token_embd') # [vocab, hidden]
    fn=get_tensor('blk.0.hc_attn_fn') # [24, 20480]
    scale=get_tensor('blk.0.hc_attn_scale')
    base=get_tensor('blk.0.hc_attn_base')
    assert emb.shape==(129280,H) and fn.shape==(MIX,HC*H) and scale.shape==(3,) and base.shape==(MIX,)
    x_np=emb[TOKEN_IDS].astype(np.float32)
    # The runtime embedding is BF16. Round source float32 back to BF16 exactly like loaded param.
    x_t=torch.tensor(x_np,dtype=torch.bfloat16)
    residual_t=x_t.unsqueeze(1).expand(-1,HC,-1).contiguous()
    fn_t=torch.tensor(fn,dtype=torch.float32)
    scale_t=torch.tensor(scale,dtype=torch.float32); base_t=torch.tensor(base,dtype=torch.float32)
    broadcast_t=fn_t.view(MIX,HC,H).sum(dim=1)
    post_t,comb_t,layer_t,pre_t=mhc_pre_delayed_torch(residual_t,broadcast_t,scale_t,base_t,RMS_EPS,HC_EPS,HC_EPS,POST_ALPHA,ITERS,x=x_t)

    residual_np=residual_t.float().numpy()
    pre_np,post_np,comb_np=np_hc_mix(residual_np,fn,scale,base)
    # Reference first-layer input uses the previous identity pre-mix [1,0,0,0].
    layer_np=residual_np[:,0]
    # deterministic fake attention result to test hc_post independently
    y_np=np.tanh(np.arange(len(TOKEN_IDS)*H,dtype=np.float32).reshape(len(TOKEN_IDS),H)/997.0-3.0)
    y_t=torch.tensor(y_np,dtype=torch.bfloat16)
    postres_t=mhc_post_torch(y_t,residual_t,post_t,comb_t).float().numpy()
    postres_np=np_hc_post(y_t.float().numpy(),residual_np,post_np,comb_np)
    # mhc_post returns BF16; round independent reference to BF16 before comparison.
    postres_ref=torch.tensor(postres_np,dtype=torch.bfloat16).float().numpy()

    # Also directly compare full-W multiply vs broadcast-W multiply for repeated stream.
    full_mix=(residual_np.reshape(len(TOKEN_IDS),-1) @ fn.T) / np.sqrt(np.mean(residual_np.reshape(len(TOKEN_IDS),-1)**2,axis=-1,keepdims=True)+RMS_EPS)
    bcast=broadcast_t.numpy()
    bcast_mix=(x_t.float().numpy() @ bcast.T) / np.sqrt(np.mean(x_t.float().numpy()**2,axis=-1,keepdims=True)+RMS_EPS)

    out={
      'status':'PASS', 'token_ids':TOKEN_IDS,
      'shapes':{'fn':list(fn.shape),'broadcast':list(bcast.shape),'residual':list(residual_np.shape)},
      'broadcast_vs_full_mix':metrics(bcast_mix,full_mix),
      'pre':metrics(pre_t.numpy(),pre_np),
      'post':metrics(post_t.squeeze(-1).numpy(),post_np),
      'comb':metrics(comb_t.numpy(),comb_np),
      'first_layer_input':metrics(layer_t.float().numpy(),layer_np),
      'hc_post_bf16':metrics(postres_t,postres_ref),
    }
    # Float32 reductions may differ by a few ulps; fail on meaningful divergence.
    # The independent Vontra tests accept rtol=2e-5/atol=1e-6 for HC coefficients.
    # The broadcast-vs-full comparison changes only float32 reduction order; gate on
    # the actual pre/post/comb outputs and BF16 post result, not bitwise accumulator order.
    if (
        out['pre']['max_abs'] > 1e-6
        or out['post']['max_abs'] > 1e-6
        or out['comb']['max_abs'] > 1e-6
        or out['first_layer_input']['max_abs'] != 0
        or out['hc_post_bf16']['max_abs'] > 2e-5
    ):
        out['status']='FAIL'
    path=ROOT/'reports/DS41-Q2-001/stage0/hc-layer0-real-node01.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if out['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
