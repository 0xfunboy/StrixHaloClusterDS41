#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
ATOL=2e-2;RTOL=2e-2;WIN=128;D=512

def corr(n,d,b,m): return d*math.log(m/(n*2*math.pi))/(2*math.log(b))
def rope_cache(maxp,c):
 d=64;s=c['rope_scaling'];f=float(s['factor']);b=float(c['compress_rope_theta']);m=int(s['original_max_position_embeddings']);lo=max(math.floor(corr(int(s['beta_fast']),d,b,m)),0);hi=min(math.ceil(corr(int(s['beta_slow']),d,b,m)),d-1);hi=hi+.001 if hi==lo else hi
 pf=b**(torch.arange(0,d,2,dtype=torch.float32)/d);ie=1/pf;ii=1/(f*pf);r=torch.clamp((torch.arange(d//2,dtype=torch.float32)-lo)/(hi-lo),0,1);inv=ii*r+ie*(1-r);t=torch.arange(maxp+1,dtype=torch.float32);fr=torch.einsum('i,j->ij',t,inv);return torch.cat((fr.cos(),fr.sin()),-1)
def rope(x,pos,cache):
 cs=cache[pos.long()];co,si=cs[:,:32],cs[:,32:];r=x[:,-64:].float().reshape(-1,32,2);e,o=r[...,0],r[...,1];ne=torch.addcmul(-o*si,e,co);no=torch.addcmul(o*co,e,si);y=x.float().clone();y[:,-64:]=torch.stack((ne,no),-1).reshape(-1,64);return y.to(torch.bfloat16)
def qat32(x):
 xf=x.float().reshape(x.shape[0],16,32);am=xf.abs().amax(-1).clamp_min(1e-4);exp=torch.ceil(torch.log2(am/448.0)).clamp(-127,127);sc=torch.pow(2.0,exp);q=torch.clamp(xf/sc[...,None],-448,448).to(torch.float8_e4m3fn);return (q.float()*sc[...,None]).reshape(-1,D).to(torch.bfloat16)
def ref(q,kv,scale,sink):
 qf=q.float();kf=kv.float();o=torch.empty_like(qf)
 for h in range(q.shape[0]):
  s=torch.mv(kf,qf[h])*float(scale);p=torch.softmax(torch.cat((s,sink[h].float().reshape(1))),0)[:-1];o[h]=torch.sum(p[:,None]*kf,0)
 return o.to(torch.bfloat16)
def metric(a,b):
 af,bf=a.float(),b.float();d=(af-bf).abs();ok=torch.isfinite(af)&torch.isfinite(bf)&(d<=ATOL+RTOL*bf.abs());return {'equivalent_gate':bool(ok.all()),'outside':int((~ok).sum()),'numel':a.numel(),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rel_l2':float((af-bf).norm())/max(float(bf.norm()),1e-30),'gate':{'atol':ATOL,'rtol':RTOL}}
def row_metric(a,b):
 af,bf=a.float(),b.float();d=(af-bf).abs();return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rel_l2':float((af-bf).norm())/max(float(bf.norm()),1e-30),'exact':bool(torch.equal(a,b))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--rank',type=int,required=True);ap.add_argument('--config',default='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix/config.json');ap.add_argument('--out',required=True);a=ap.parse_args();p=torch.load(a.input,weights_only=False,map_location='cpu');cfg=json.load(open(a.config))['text_config'];cache=rope_cache(1587,cfg);rows=[]
 for ck in ('0','1'):
  pr=p['layer2_projection'][ck];at=p['layer2_attention'][ck];pos=pr['positions'].long();src=pr['kv_current_chunk'].cpu();n=int(at['N']);seq=int(at['seq_lens'][0]);gl=int(at['gather_lens'][0]);idx=at['valid_context_indices'].long().cpu();swa=idx[-WIN:];logical=seq-gl+(swa-n);si=logical-int(pos[0]);assert torch.equal(si,torch.arange(len(pos)-WIN,len(pos))) and torch.equal(pos[si],logical)
  pre=src[si];roped=rope(pre,logical,cache);official=qat32(roped);local=at['context_rows'].cpu()[-WIN:];ctx=at['context_rows'].cpu().clone();ctx[-WIN:]=official
  ol=ref(at['q_final'].cpu(),at['context_rows'].cpu(),float(at['scale']),at['attn_sink'].cpu());oa=ref(at['q_final'].cpu(),ctx,float(at['scale']),at['attn_sink'].cpu());san=metric(at['kernel_output_final'].cpu(),ol);delta=metric(oa,ol)
  rows.append({'chunk':int(ck),'logical_swa':[int(logical[0]),int(logical[-1])],'row_delta_official_qat_vs_local':row_metric(official,local),'local_reference_sanity':san,'attention_delta_official_window_qat_vs_local':delta})
 status='MATERIAL_COMPONENT_EFFECT' if any(not r['attention_delta_official_window_qat_vs_local']['equivalent_gate'] for r in rows) else 'EQUIVALENT_AT_ATTENTION_GATE'
 out={'schema':'ds41-r2-window-qat-differential-v1','status':status,'rank':a.rank,'input':a.input,'prompt_tokens':p.get('prompt_tokens'),'request_index':p.get('request_index'),'change':'replace only final-query 128 SWA rows with V4.1 reference window FP8 block32 UE8M0 QDQ; Q, compressed rows, indices, sink and attention math fixed','gate':{'atol':ATOL,'rtol':RTOL,'basis':'same upstream sparse-attention equivalence gate used in prior saved-packet reference'},'chunks':rows,'scope':'component effect only; not correctness or full-model quality','execution':{'cpu_only':True,'model_load':False,'generation':False}}
 Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
