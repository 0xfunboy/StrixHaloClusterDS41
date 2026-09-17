#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np
import torch
import gguf

BF16_RTOL=0.016; BF16_ATOL=1e-5
H=5120; QRA=1280; HD=512; HEADS=64; ROPE=64; TP=2
EPS=1e-20

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def bf16_ulp(a,b):
 def key(t):
  u=t.contiguous().view(torch.int16).to(torch.int64)&0xFFFF
  return torch.where(u>=0x8000,0xFFFF-u,u+0x8000)
 return (key(a)-key(b)).abs()

def metric(actual,ref,row_shape=None):
 a=actual.float(); r=ref.float(); d=(a-r).abs(); close=torch.isclose(a,r,rtol=BF16_RTOL,atol=BF16_ATOL)
 rel=float((a-r).norm())/max(float(r.norm()),1e-30); flat=int(d.reshape(-1).argmax())
 coord=[int(v) for v in np.unravel_index(flat,d.shape)]; ulp=bf16_ulp(actual,ref)
 out={'finite':bool(torch.isfinite(a).all() and torch.isfinite(r).all()),'allclose':bool(close.all()),'outside':int((~close).sum()),'numel':a.numel(),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rel_l2':rel,'worst_coord':coord,'reference':float(r[tuple(coord)]),'actual':float(a[tuple(coord)]),'worst_ulp':int(ulp.max())}
 if row_shape is not None:
  rr=d.reshape(row_shape[0],-1); cr=close.reshape(row_shape[0],-1); ur=ulp.reshape(row_shape[0],-1)
  out['per_row']=[{'row':i,'max_abs':float(rr[i].max()),'outside':int((~cr[i]).sum()),'worst_ulp':int(ur[i].max()),'ref_rms':float(r.reshape(row_shape[0],-1)[i].square().mean().sqrt()),'actual_rms':float(a.reshape(row_shape[0],-1)[i].square().mean().sqrt())} for i in range(row_shape[0])]
 return out

def rms_ref(x,w):
 xf=x.float(); return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)*w.float()).to(torch.bfloat16)

def yarn_range(lowrot,highrot,dim,base,maxpos):
 f=lambda n:(dim*math.log(maxpos/(n*2*math.pi)))/(2*math.log(base))
 return max(math.floor(f(lowrot)),0), min(math.ceil(f(highrot)),dim-1)

def rope_cache(maxpos,cfg):
 sc=cfg['rope_scaling']; factor=float(sc['factor']); base=float(cfg['compress_rope_theta']); orig=int(sc['original_max_position_embeddings']); dim=int(cfg['qk_rope_head_dim'])
 pf=base**(torch.arange(0,dim,2,dtype=torch.float32)/dim); ie=1/pf; ii=1/(factor*pf); lo,hi=yarn_range(int(sc['beta_fast']),int(sc['beta_slow']),dim,base,orig)
 ramp=torch.clamp((torch.arange(dim//2,dtype=torch.float32)-lo)/(hi-lo if hi!=lo else .001),0,1); mask=(1-ramp)
 inv=ii*(1-mask)+ie*mask; t=torch.arange(maxpos+1,dtype=torch.float32); f=torch.einsum('i,j->ij',t,inv)
 return torch.cat((f.cos(),f.sin()),-1)

def apply_rope(x,pos,cache):
 half=ROPE//2; cs=cache[pos.long()]; cos,sin=cs[...,:half],cs[...,half:]; rp=x[...,-ROPE:].float(); sh=rp.shape; rp=rp.reshape(*sh[:-1],half,2); e,o=rp[...,0],rp[...,1]
 ne=torch.addcmul(-o*sin,e,cos); no=torch.addcmul(o*cos,e,sin); out=x.clone().float(); out[...,-ROPE:]=torch.stack((ne,no),-1).reshape(sh); return out.to(x.dtype)

class Weights:
 def __init__(self,root):
  self.readers=[gguf.GGUFReader(str(p),'r') for p in sorted(Path(root).glob('*.gguf'))]
 def get(self,name):
  for r in self.readers:
   for t in r.tensors:
    if t.name==name:
     if t.tensor_type.name!='BF16': raise RuntimeError(f'{name} not BF16: {t.tensor_type.name}')
     a=t.data.view(np.uint16).reshape(*(int(x) for x in reversed(t.shape))).copy()
     return torch.from_numpy(a).view(torch.bfloat16), hashlib.sha256(t.data.view(np.uint8).tobytes()).hexdigest()
  raise KeyError(name)

def analyze(packet,rank,model):
 cfg=json.load(open(Path(model)/'config.json'))['text_config']; wc=Weights(model)
 wqa,sqa=wc.get('blk.2.attn_q_a'); wkv,skv=wc.get('blk.2.attn_kv'); qnw,sqn=wc.get('blk.2.attn_q_a_norm'); kvnw,skn=wc.get('blk.2.attn_kv_a_norm'); wqb,sqb=wc.get('blk.2.attn_q_b')
 if wqa.shape!=(QRA,H) or wkv.shape!=(HD,H) or wqb.shape!=(HEADS*HD,QRA): raise RuntimeError('weight shape mismatch')
 qrows=(HEADS*HD)//TP; qb=wqb[rank*qrows:(rank+1)*qrows]
 rc=rope_cache(max(int(v['positions'][-1]) for v in packet['layer2_projection'].values()),cfg)
 chunks=[]
 for ck in sorted(packet['layer2_full'],key=int):
  x=packet['layer2_full'][ck]['attn_norm']['x'].cpu(); pos=packet['layer2_projection'][ck]['positions'].long().cpu(); kvsnap=packet['layer2_projection'][ck]['kv_current_chunk'].cpu(); qsnap=packet['layer2_attention'][ck]['q_final'].cpu()
  qpos=int(packet['layer2_attention'][ck]['q_position'][0]); swa=min(qpos+1,128); logical=torch.arange(qpos-swa+1,qpos+1,dtype=torch.long); idx=logical-int(pos[0])
  if not torch.equal(pos.index_select(0,idx),logical): raise RuntimeError(f'chunk{ck} row mapping mismatch')
  xs=x.index_select(0,idx); qx=x[-1:]
  kvraw=(xs.float()@wkv.float().T).to(torch.bfloat16); kvref=rms_ref(kvraw,kvnw); kva=kvsnap.index_select(0,idx)
  qaraw=(qx.float()@wqa.float().T).to(torch.bfloat16); qarn=rms_ref(qaraw,qnw); qpre=(qarn.float()@qb.float().T).to(torch.bfloat16).reshape(1,HEADS//TP,HD); qref=apply_rope(qpre,torch.tensor([qpos]),rc)[0]
  km=metric(kva,kvref,row_shape=(swa,HD)); qm=metric(qsnap,qref,row_shape=(HEADS//TP,HD))
  chunks.append({'chunk':int(ck),'positions':[int(pos[0]),int(pos[-1])],'kv_rows_logical':[int(logical[0]),int(logical[-1])],'q_position':qpos,'kv':km,'q':qm,'status':'PASS' if km['finite'] and km['allclose'] and qm['finite'] and qm['allclose'] else 'FAIL'})
 status='PASS' if all(c['status']=='PASS' for c in chunks) else 'FAIL'
 return {'schema':'ds41-qkv-projection-reference-v1','status':status,'rank':rank,'prompt_tokens':packet.get('prompt_tokens'),'request_index':packet.get('request_index'),'request_id':packet.get('request_id'),'gates':{'dtype':'bfloat16','rtol':BF16_RTOL,'atol':BF16_ATOL,'source':'torch.testing default_tolerances(torch.bfloat16); fixed before fixture result'},'weights':{'wq_a':sqa,'wkv':skv,'q_norm':sqn,'kv_norm':skn,'wq_b':sqb,'q_b_rows':[rank*qrows,(rank+1)*qrows-1]},'contract':{'input':'layer2_full[chunk].attn_norm.x','kv_endpoint':'qkv projection -> kv RMSNorm; compare pre-RoPE kv_current_chunk','q_endpoint':'q_a projection -> q RMSNorm -> rank-local q_b -> compressed YaRN RoPE; compare q_final','accumulation_reference':'explicit FP32 matmul, BF16 cast at runtime linear boundaries; explicit FP32 RMSNorm -> BF16','no_runtime_ops_reused':True},'chunks':chunks,'model_loaded':False,'generation_requests':0,'gpu_kernel_launched':False,'scope_limit':'saved code2k1588 layer2 endpoints only; input hidden-state correctness and downstream layers/full retrieval not certified'}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--model',default='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix'); ap.add_argument('--output',required=True); a=ap.parse_args()
 p=torch.load(a.input,weights_only=False,map_location='cpu');
 if p.get('prompt_tokens')!=1588 or p.get('request_index')!=0: out={'schema':'ds41-qkv-projection-reference-v1','status':'INPUT_INCOMPLETE','reason':'not frozen code2k1588 request0'}
 else: out=analyze(p,a.rank,a.model); out['input']=a.input; out['input_sha256']=sha(a.input); out['model_config_sha256']=sha(Path(a.model)/'config.json')
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); return 0 if out['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
