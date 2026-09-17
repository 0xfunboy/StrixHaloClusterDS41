#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import torch
NOPE,ROPE,HEAD,QBLOCK,FP8MAX,WINDOW=448,64,512,64,448.0,128

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def corr_dim(rot,dim,base,maxp): return dim*math.log(maxp/(rot*2*math.pi))/(2*math.log(base))
def rope_cache(maxp,cfg):
 sc=cfg['rope_scaling']; factor=float(sc['factor']); base=float(cfg['compress_rope_theta']); dim=int(cfg['qk_rope_head_dim']); orig=int(sc['original_max_position_embeddings'])
 lo=max(math.floor(corr_dim(int(sc['beta_fast']),dim,base,orig)),0); hi=min(math.ceil(corr_dim(int(sc['beta_slow']),dim,base,orig)),dim-1)
 if lo==hi: hi+=.001
 ramp=torch.clamp((torch.arange(dim//2,dtype=torch.float32)-lo)/(hi-lo),0,1)
 posfreq=base**(torch.arange(0,dim,2,dtype=torch.float32)/dim); mask=1-ramp
 inv=(1/(factor*posfreq))*(1-mask)+(1/posfreq)*mask
 t=torch.arange(maxp+1,dtype=torch.float32); fr=torch.einsum('i,j->ij',t,inv)
 return torch.cat((fr.cos(),fr.sin()),dim=-1)
def apply_rope(x,pos,cache):
 half=ROPE//2; cs=cache[pos.long()].float(); cos,sin=cs[...,:half],cs[...,half:]; r=x[...,-ROPE:].float(); sh=r.shape; r=r.reshape(*sh[:-1],half,2); e,o=r[...,0],r[...,1]
 ne=torch.addcmul(-o*sin,e,cos); no=torch.addcmul(o*cos,e,sin); out=x.clone().float(); out[...,-ROPE:]=torch.stack((ne,no),-1).reshape(sh); return out.to(x.dtype)
def ulp(a,b):
 def k(t):
  u=t.contiguous().view(torch.int16).long()&0xffff; return torch.where(u>=0x8000,0xffff-u,u+0x8000)
 return (k(a)-k(b)).abs()
def scales(x):
 blk=x.view(x.shape[0],NOPE//QBLOCK,QBLOCK); a=blk.abs().amax(-1).clamp(min=1e-4); return torch.pow(2.,torch.ceil(torch.log2(a/FP8MAX)))
def chunk_result(k,p,cache):
 pr=p['layer2_projection'][k]; at=p['layer2_attention'][k]; pos=pr['positions'].long(); srcall=pr['kv_current_chunk'].cpu(); ctx=at['context_rows'].cpu(); idx=at['valid_context_indices'].long().cpu()
 N,M=int(at['N']),int(at['M']); seq,gather=int(at['seq_lens'][0]),int(at['gather_lens'][0]); qpos=int(at['q_position'][0]); clen=int(at['combined_lens_final'][0]); top=min((qpos+1)//2,512); sw=min(qpos+1,WINDOW)
 assoc={'pass':False}
 if sw!=WINDOW or len(idx)!=clen or len(ctx)!=clen: return {'chunk':int(k),'status':'REFERENCE_INPUT_INCOMPLETE','association':assoc,'reason':'unexpected lengths'}
 wi=idx[-sw:]; local=wi-N; logical=seq-gather+local; si=logical-int(pos[0]); exp_si=torch.arange(len(pos)-sw,len(pos))
 checks={'combined_len':clen==top+sw,'workspace_region':bool(torch.all((wi>=N)&(wi<M))),'logical_consecutive':bool(torch.equal(logical,torch.arange(int(logical[0]),int(logical[-1])+1))),'source_tail':bool(torch.equal(si,exp_si)),'source_positions':bool(torch.equal(pos.index_select(0,si),logical))}
 assoc={'pass':all(checks.values()),'checks':checks,'projection_positions':[int(pos[0]),int(pos[-1])],'logical_swa':[int(logical[0]),int(logical[-1])],'source_indices':[int(si[0]),int(si[-1])],'workspace_indices':[int(wi[0]),int(wi[-1])],'gather_start':seq-gather,'gather_len':gather,'topk_len':top,'swa_len':sw}
 if not assoc['pass']: return {'chunk':int(k),'status':'REFERENCE_INPUT_INCOMPLETE','association':assoc}
 src=srcall.index_select(0,si); actual=ctx[-sw:]
 if tuple(src.shape)!=(WINDOW,HEAD) or src.shape!=actual.shape: return {'chunk':int(k),'status':'REFERENCE_INPUT_INCOMPLETE','association':assoc,'reason':'shape mismatch'}
 finite=bool(torch.isfinite(src.float()).all() and torch.isfinite(actual.float()).all()); ref=apply_rope(src,logical,cache)
 nd=(actual[:,:NOPE].float()-ref[:,:NOPE].float()).abs(); sc=scales(ref[:,:NOPE].float()); td=nd.amax(-1); bound=16*sc.amax(-1); npass=td<=bound
 bd=nd.view(WINDOW,NOPE//QBLOCK,QBLOCK).amax(-1); bb=16*sc; bpass=bd<=bb; ratio=td/bound.clamp_min(1e-30); wt=int(ratio.argmax()); flat=int(nd.reshape(-1).argmax()); nr,nc=divmod(flat,NOPE)
 ra,rr=actual[:,NOPE:],ref[:,NOPE:]; u=ulp(ra,rr); tu=u.amax(-1); rpass=tu<=1; uf=int(u.reshape(-1).argmax()); ur,uc=divmod(uf,ROPE); ad=(ra.float()-rr.float()).abs()
 status='PASS' if finite and bool(npass.all()) and bool(rpass.all()) else 'FAIL'
 return {'chunk':int(k),'status':status,'association':assoc,'finite':finite,'capture_boundary':'kv_current_chunk is post-RMS/pre-cache-insert/pre-RoPE; SWA context_rows is post-RoPE/post-UE8M0 gather-dequant','reference_transform':'V4.1 compressed YaRN RoPE exactly once on last64; NoPE448 unchanged before cache quantization','nope448':{'tokens_outside_gate':int((~npass).sum()),'worst_token_ratio':float(ratio[wt]),'worst_token_logical_position':int(logical[wt]),'worst_token_max_abs':float(td[wt]),'worst_token_bound':float(bound[wt]),'global_max_abs':float(nd.max()),'global_max_coordinate':{'logical_position':int(logical[nr]),'dim':nc,'reference':float(ref[nr,nc]),'actual':float(actual[nr,nc])},'diagnostic_blocks_outside_16x_own_scale':int((~bpass).sum()),'per_token':[{'logical_position':int(logical[i]),'max_abs':float(td[i]),'max_scale':float(sc[i].max()),'bound':float(bound[i]),'pass':bool(npass[i]),'blocks':[{'block':j,'max_abs':float(bd[i,j]),'scale':float(sc[i,j]),'bound':float(bb[i,j]),'within_bound':bool(bpass[i,j])} for j in range(sc.shape[1])]} for i in range(WINDOW)]},'rope64':{'tokens_outside_gate':int((~rpass).sum()),'worst_ulp':int(u.max()),'worst_coordinate':{'logical_position':int(logical[ur]),'dim':NOPE+uc,'reference':float(rr[ur,uc]),'actual':float(ra[ur,uc]),'abs_diff':float(ad[ur,uc])},'per_token':[{'logical_position':int(logical[i]),'max_ulp':int(tu[i]),'pass':bool(rpass[i])} for i in range(WINDOW)]}}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--output',required=True); ap.add_argument('--config',default='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix/config.json'); a=ap.parse_args()
 p=Path(a.input); cp=Path(a.config); out=Path(a.output); pkt=torch.load(p,weights_only=False,map_location='cpu'); cfg=json.loads(cp.read_text())['text_config']
 if pkt.get('prompt_tokens')!=1588 or pkt.get('request_index')!=0: res={'schema':'ds41-swa-recent-kv-reference-v1','status':'REFERENCE_INPUT_INCOMPLETE','reason':'not request0/prompt1588','input':str(p)}
 elif int(cfg['compress_ratios'][2])!=2 or int(cfg['sliding_window'])!=WINDOW: res={'schema':'ds41-swa-recent-kv-reference-v1','status':'REFERENCE_INPUT_INCOMPLETE','reason':'layer2 config mismatch','input':str(p)}
 else:
  cache=rope_cache(max(int(v['positions'][-1]) for v in pkt['layer2_projection'].values()),cfg); chunks=[chunk_result(k,pkt,cache) for k in sorted(pkt['layer2_attention'],key=int)]; st='PASS' if all(x['status']=='PASS' for x in chunks) else ('REFERENCE_INPUT_INCOMPLETE' if any(x['status']=='REFERENCE_INPUT_INCOMPLETE' for x in chunks) else 'FAIL')
  res={'schema':'ds41-swa-recent-kv-reference-v1','status':st,'rank':a.rank,'input':str(p),'input_sha256':sha(p),'request_index':pkt.get('request_index'),'request_id':pkt.get('request_id'),'prompt_tokens':pkt.get('prompt_tokens'),'config':str(cp),'config_sha256':sha(cp),'layer':2,'compress_ratio':2,'sliding_window':WINDOW,'rope':{'type':'deepseek_yarn','base':float(cfg['compress_rope_theta']),'factor':float(cfg['rope_scaling']['factor']),'original_max_position_embeddings':int(cfg['rope_scaling']['original_max_position_embeddings']),'beta_fast':int(cfg['rope_scaling']['beta_fast']),'beta_slow':int(cfg['rope_scaling']['beta_slow']),'mscale':0,'mscale_all_dim':0},'gates':{'nope448':'per-token max_abs <=16*max UE8M0 scale, FP8_MAX=448','rope64':'max BF16 ULP <=1'},'chunks':chunks,'model_loaded':False,'generation_requests':0,'gpu_kernel_launched':False,'scope_limit':'saved code2k1588 layer2 SWA rows only; not discriminator1571/full-model quality'}
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2)); return 0 if res['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
