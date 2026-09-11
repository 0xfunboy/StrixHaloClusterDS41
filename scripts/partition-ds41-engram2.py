#!/usr/bin/env python3
"""Split verified V4.1 2-bit Engram SafeTensors into TP-rank-local row ranges."""
from __future__ import annotations
import argparse,hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.ds41.affine_safetensors import SafeTensorMMap

LAYER_FILE={1:'model-00047-of-00048.safetensors',14:'model-00048-of-00048.safetensors'}
SOURCE_SHA={
  1:'e80b0d1481bb8e32196398d5dd670df721604517ecdfdf279485cc92ca6a19d2',
  14:'1aee6ffb59d1314d3264c95d61109ec18df61aa48101073b45f1c6730e8d9230',
}

def is_prime(n):
    if n<2:return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n%p==0:return n==p
    d=n-1;r=0
    while d%2==0:r+=1;d//=2
    for a in (2,7,61):
        x=pow(a,d,n)
        if x in (1,n-1):continue
        for _ in range(r-1):
            x=x*x%n
            if x==n-1:break
        else:return False
    return True

def next_prime(start,seen):
    x=start+1
    while x in seen or not is_prime(x):x+=1
    seen.add(x);return x

def layouts(c):
    seen=set();out={}
    for layer,expected in zip(c['engram_layer_ids'],c['engram_num_embeddings']):
        sizes=[]
        for _ in range(c['engram_max_ngram_size']-1):
            cur=c['engram_vocab_size']-1
            for _ in range(c['engram_n_heads']):
                cur=next_prime(cur,seen);sizes.append(cur)
        assert sum(sizes)==expected,(layer,sum(sizes),expected)
        out[layer]=sizes
    return out

def copy_span(src_fd,dst_fd,offset,count,chunk=16<<20):
    left=count;pos=offset
    while left:
        n=min(left,chunk);data=os.pread(src_fd,n,pos)
        if len(data)!=n:raise IOError(f'truncated source at {pos}')
        view=memoryview(data)
        while view:
            wrote=os.write(dst_fd,view);view=view[wrote:]
        pos+=n;left-=n

def build_one(src:Path,dst:Path,layer:int,row0:int,row1:int,rank:int,tp:int):
    st=SafeTensorMMap(src);names=list(st.entries)
    specs=[];off=0
    for name in names:
        e=st.entries[name];shape=list(e.shape);src_off=e.offset;nbytes=e.nbytes
        if name.startswith(f'layers.{layer}.engram.embed.'):
            stride=e.nbytes//e.shape[0];src_off=e.offset+row0*stride;nbytes=(row1-row0)*stride;shape[0]=row1-row0
        specs.append((name,e.dtype,shape,src_off,nbytes,off));off+=nbytes
    meta={
      'format':'mlx','ds41_tp_rank':str(rank),'ds41_tp_size':str(tp),
      'ds41_vocab_start':str(row0),'ds41_vocab_end':str(row1),
      'ds41_source_sha256':SOURCE_SHA[layer],
    }
    header={name:{'dtype':dtype,'shape':shape,'data_offsets':[dst_off,dst_off+nbytes]}
            for name,dtype,shape,src_off,nbytes,dst_off in specs}
    header['__metadata__']=meta
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((8-len(raw)%8)%8)
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.part')
    with open(tmp,'wb',buffering=0) as f:
        f.write(struct.pack('<Q',len(raw)));f.write(raw)
        sfd=os.open(src,os.O_RDONLY)
        try:
            for _,_,_,src_off,nbytes,_ in specs:copy_span(sfd,f.fileno(),src_off,nbytes)
        finally:os.close(sfd)
        os.fsync(f.fileno())
    os.replace(tmp,dst);st.close()
    verify=SafeTensorMMap(dst)
    emb=verify.entries[f'layers.{layer}.engram.embed.weight']
    assert emb.shape[0]==row1-row0
    verify.close()
    h=hashlib.sha256()
    with open(dst,'rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return dst.stat().st_size,h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output-root',required=True);ap.add_argument('--tp',type=int,default=2);a=ap.parse_args()
    cfg=json.load(open(a.config))['text_config'];lay=layouts(cfg);receipt={'tp':a.tp,'layers':{}}
    for layer in cfg['engram_layer_ids']:
        sizes=lay[layer];part=math.ceil(len(sizes)/a.tp);receipt['layers'][str(layer)]={}
        for rank in range(a.tp):
            hs=rank*part;he=min(hs+part,len(sizes));r0=sum(sizes[:hs]);r1=sum(sizes[:he])
            dst=Path(a.output_root)/f'rank{rank}'/LAYER_FILE[layer]
            size,sha=build_one(Path(a.source)/LAYER_FILE[layer],dst,layer,r0,r1,rank,a.tp)
            receipt['layers'][str(layer)][str(rank)]={'vocab_start':r0,'vocab_end':r1,'bytes':size,'sha256':sha,'path':str(dst)}
            print(json.dumps({'layer':layer,'rank':rank,'rows':r1-r0,'bytes':size,'sha256':sha}),flush=True)
    out=Path(a.output_root)/'partition-receipt.json';out.write_text(json.dumps(receipt,indent=2)+'\n');print(out)
if __name__=='__main__':main()
