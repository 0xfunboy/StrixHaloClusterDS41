#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np, torch
REL=Path('/home/funboy/.local/share/haloclu-ds41/releases/k2-layer2diag-d454801')
sys.path.insert(0,str(REL/'.vendor/llama-v41/gguf-py'))
import gguf
MODEL=Path('/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix')
EPS=1e-20; HC_EPS=1e-6; HC=4; H=5120; ROPE=64; HEAD=512; GROUPS=8; LOCAL_GROUPS=4; ORANK=1024

def sha_bytes(x): return hashlib.sha256(x).hexdigest()
def tensor(name):
 for p in sorted(MODEL.glob('*.gguf')):
  r=gguf.GGUFReader(str(p))
  for t in r.tensors:
   if t.name==name: return t
 raise KeyError(name)
def bf16(name):
 t=tensor(name); assert t.tensor_type.name=='BF16'; a=t.data.view(np.uint16).reshape(*(int(x) for x in reversed(t.shape))).copy(); return torch.from_numpy(a).view(torch.bfloat16)
def f32(name):
 t=tensor(name); assert t.tensor_type.name=='F32'; return torch.from_numpy(np.asarray(t.data).view(np.float32).reshape(*(int(x) for x in reversed(t.shape))).copy())
def ulp(a,b):
 def key(t):
  u=t.contiguous().view(torch.int16).to(torch.int64)&0xffff; return torch.where(u>=0x8000,0xffff-u,u+0x8000)
 return (key(a)-key(b)).abs()
def close_metric(a,r,atol,rtol):
 af,rf=a.float(),r.float(); d=(af-rf).abs(); ok=torch.isclose(af,rf,atol=atol,rtol=rtol); i=int(d.reshape(-1).argmax()); c=[int(x) for x in np.unravel_index(i,d.shape)]
 return {'pass':bool(ok.all() and torch.isfinite(af).all() and torch.isfinite(rf).all()),'outside':int((~ok).sum()),'numel':a.numel(),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rel_l2':float((af-rf).norm())/max(float(rf.norm()),1e-30),'worst_ulp':int(ulp(a,r).max()) if a.dtype==torch.bfloat16 and r.dtype==torch.bfloat16 else None,'worst_coord':c,'reference':float(rf[tuple(c)]),'actual':float(af[tuple(c)]),'gate':{'atol':atol,'rtol':rtol}}
def relmax_metric(a,r,rel_gate,max_gate):
 af,rf=a.float(),r.float(); d=(af-rf).abs(); rel=float((af-rf).norm())/max(float(rf.norm()),1e-30); i=int(d.reshape(-1).argmax()); c=[int(x) for x in np.unravel_index(i,d.shape)]
 return {'pass':bool(torch.isfinite(af).all() and torch.isfinite(rf).all() and rel<=rel_gate and float(d.max())<=max_gate),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rel_l2':rel,'worst_coord':c,'reference':float(rf[tuple(c)]),'actual':float(af[tuple(c)]),'gate':{'rel_l2_max':rel_gate,'max_abs':max_gate}}
def rms(x,w):
 xf=x.float(); return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)*w.float()).to(torch.bfloat16)
def mhc_chunk(pack,ck,fn,sc,base,nw):
 e=pack['layer2_full'][ck]['entry']; a=pack['layer2_full'][ck]['attn_mhc_pre']; an=pack['layer2_full'][ck]['attn_norm']['x']
 x=e['x']; res=e['residual']; pm=e['post_mix']; rm=e['res_mix']; carried=e['pre_mix']; assert all(v is not None for v in (res,pm,rm,carried))
 mixed=torch.einsum('tij,tih->tjh',rm.float(),res.float()); post=pm.float()*x.unsqueeze(1).float(); res1=(mixed+post).to(torch.bfloat16)
 xf=res1.flatten(1).float(); mixes=(xf@fn.float().T)*torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)
 pre=torch.sigmoid(mixes[:,:HC]*sc[0]+base[:HC])+HC_EPS
 postn=torch.sigmoid(mixes[:,HC:2*HC]*sc[1]+base[HC:2*HC])*2.0
 comb=mixes[:,2*HC:].view(-1,HC,HC)*sc[2]+base[2*HC:].view(1,HC,HC); comb=torch.softmax(comb,dim=-1)+HC_EPS; comb=comb/(comb.sum(-2,keepdim=True)+HC_EPS)
 for _ in range(19): comb=comb/(comb.sum(-1,keepdim=True)+HC_EPS); comb=comb/(comb.sum(-2,keepdim=True)+HC_EPS)
 layer=(carried.unsqueeze(-1)*res1.float()).sum(1).to(torch.bfloat16); norm=rms(layer,nw)
 m={'residual_after_prev_post':close_metric(a['residual'],res1,0.016,0.01),'layer_input_x':close_metric(a['x'],layer,0.016,0.01),'new_pre_mix':close_metric(a['pre_mix'],pre,1e-5,1e-4),'new_post_mix':close_metric(a['post_mix'].squeeze(-1),postn,1e-5,1e-4),'new_res_mix':close_metric(a['res_mix'],comb,1e-5,1e-4),'attn_norm_x':close_metric(an,norm,1e-5,0.016)}
 return {'status':'PASS' if all(v['pass'] for v in m.values()) else 'FAIL','metrics':m,'rows':int(x.shape[0])}
def yarn_cache(maxpos,cfg):
 d=64; s=cfg['rope_scaling']; factor=float(s['factor']); base=float(cfg['compress_rope_theta']); orig=int(s['original_max_position_embeddings']); bf=int(s['beta_fast']); bs=int(s['beta_slow'])
 pf=base**(torch.arange(0,d,2,dtype=torch.float32)/d); ie=1/pf; ii=1/(factor*pf)
 def corr(n): return d*math.log(orig/(n*2*math.pi))/(2*math.log(base))
 lo=max(math.floor(corr(bf)),0); hi=min(math.ceil(corr(bs)),d-1); hi=hi+0.001 if hi==lo else hi; ramp=torch.clamp((torch.arange(d//2,dtype=torch.float32)-lo)/(hi-lo),0,1); mask=1-ramp; inv=ii*(1-mask)+ie*mask
 t=torch.arange(maxpos+1,dtype=torch.float32); f=torch.einsum('i,j->ij',t,inv); return torch.cat((f.cos(),f.sin()),-1)
def inv_rope(x,pos,cache):
 cs=cache[pos].float(); co,si=cs[:32],-cs[32:]; rope=x[...,-ROPE:].float().reshape(*x.shape[:-1],32,2); even,odd=rope[...,0],rope[...,1]; ne=torch.addcmul(-odd*si,even,co); no=torch.addcmul(odd*co,even,si); out=x.clone().float(); out[...,-ROPE:]=torch.stack((ne,no),-1).reshape(*x.shape[:-1],ROPE); return out.to(x.dtype)
def oproj_local(pack,ck,rank,woa,wob,cache):
 a=pack['layer2_attention'][ck]; qpos=int(a['q_position'][0]); o=a['kernel_output_final'].cpu(); assert tuple(o.shape)==(32,512)
 inv=inv_rope(o,qpos,cache).view(LOCAL_GROUPS,-1)
 row0=rank*4096; wa=woa[row0:row0+4096].reshape(LOCAL_GROUPS,ORANK,4096)
 z=torch.einsum('gd,grd->gr',inv.float(),wa.float()).to(torch.bfloat16).flatten()
 wb=wob[:,row0:row0+4096]; local=(z.float()@wb.float().T).to(torch.bfloat16)
 target=pack['layer2_full'][ck]['attention_out']['x'][-1].cpu()
 return {'qpos':qpos,'local':local,'target':target,'kernel_output_shape':list(o.shape),'z_shape':list(z.shape)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--packet',required=True); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--tensor-out',required=True); q=ap.parse_args(); pack=torch.load(q.packet,weights_only=False,map_location='cpu'); assert pack['prompt_tokens']==1588 and pack['request_index']==0
 fn,sc,base,nw=f32('blk.2.hc_attn_fn'),f32('blk.2.hc_attn_scale'),f32('blk.2.hc_attn_base'),bf16('blk.2.attn_norm'); woa,wob=bf16('blk.2.attn_output_a'),bf16('blk.2.attn_output_b')
 cfg=json.load(open(MODEL/'config.json'))['text_config']; cache=yarn_cache(1587,cfg); chunks={}; tensors={}
 for ck in ('0','1'):
  m=mhc_chunk(pack,ck,fn,sc,base,nw); o=oproj_local(pack,ck,q.rank,woa,wob,cache); chunks[ck]={'mhc':m,'oproj_local':{'qpos':o['qpos'],'kernel_output_shape':o['kernel_output_shape'],'z_shape':o['z_shape']}}; tensors[ck]={'local':o['local'],'target':o['target']}
 status='PASS' if all(chunks[k]['mhc']['status']=='PASS' for k in chunks) else 'FAIL'
 out={'schema':'ds41-recovery-r1b-rank-v1','status':status,'rank':q.rank,'packet':q.packet,'prompt_tokens':1588,'request_index':0,'chunks':chunks,'weights':{'mhc':'blk.2.hc_attn_{fn,scale,base}+attn_norm','woa':'blk.2.attn_output_a BF16','wob':'blk.2.attn_output_b BF16'},'execution':{'cpu_only':True,'model_load':False,'gpu':False}}
 Path(q.out).write_text(json.dumps(out,indent=2)+'\n'); torch.save(tensors,q.tensor_out); print(json.dumps({'status':status,'rank':q.rank,'mhc':{k:chunks[k]['mhc']['status'] for k in chunks},'qpos':{k:chunks[k]['oproj_local']['qpos'] for k in chunks}},indent=2)); return 0 if status=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
