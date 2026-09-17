#!/usr/bin/env python3
from __future__ import annotations
import json,os,struct,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from runtime.ds41.affine_safetensors import float32_to_bf16_words


def pack2(q):
    q=np.asarray(q,dtype=np.uint32);x=q.reshape(*q.shape[:-1],-1,16)
    shifts=np.arange(0,32,2,dtype=np.uint32)
    return np.bitwise_or.reduce(x<<shifts,axis=-1).astype('<u4')


def bf16(x): return float32_to_bf16_words(np.asarray(x,dtype=np.float32))


def write(path:Path, items):
    header={};blobs=[];off=0
    for name,dtype,arr in items:
        arr=np.ascontiguousarray(arr);raw=arr.tobytes()
        header[name]={'dtype':dtype,'shape':list(arr.shape),'data_offsets':[off,off+len(raw)]}
        off+=len(raw);blobs.append(raw)
    header['__metadata__']={'format':'mlx'}
    h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((8-len(h)%8)%8)
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(h)));f.write(h)
        for b in blobs:f.write(b)


def main():
    # 3 local hash heads, six total rows, each row 256 dims affine2.
    rows=6; emb_q=(np.arange(rows*256,dtype=np.uint32).reshape(rows,256)%4)
    emb_s=np.full((rows,4),.5,np.float32);emb_b=np.full((rows,4),-1,np.float32)
    hidden=8;hc=1;heads=3;inp=heads*256;out=hidden*(hc+1)
    wq=(np.arange(out*inp,dtype=np.uint32).reshape(out,inp)%4)
    ws=np.full((out,inp//64),.25,np.float32);wb=np.full_like(ws,-.5)
    q=np.arange(hc*hidden,dtype=np.float32).reshape(hc,hidden)/8
    k=np.arange(hc*hidden,dtype=np.float32).reshape(hc,hidden)/16
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        write(root/'model-00047-of-00048.safetensors',[
            ('layers.1.engram.embed.weight','U32',pack2(emb_q)),
            ('layers.1.engram.embed.scales','BF16',bf16(emb_s)),
            ('layers.1.engram.embed.biases','BF16',bf16(emb_b)),
            ('layers.1.engram.q_weight','BF16',bf16(q)),
            ('layers.1.engram.k_weight','BF16',bf16(k)),
            ('layers.1.engram.wkv.weight','U32',pack2(wq)),
            ('layers.1.engram.wkv.scales','BF16',bf16(ws)),
            ('layers.1.engram.wkv.biases','BF16',bf16(wb)),
        ])
        import vllm.models.deepseek_v4_1.common.disk_engram as de
        de.get_tensor_model_parallel_world_size=lambda:1
        de.get_tensor_model_parallel_rank=lambda:0
        src,emb,qt,kt,wkv=de.load_disk_engram_sidecar(
            root=root,layer_id=1,num_embeddings=rows,dim=hidden,hc_mult=hc,
            head_sizes=(2,2,2),device=torch.device('cpu'))
        ids=torch.tensor([[0,2,4],[1,3,5]],dtype=torch.int64)
        out=torch.empty((2,3,256),dtype=torch.bfloat16);emb.lookup(ids,out)
        expected=emb_q.astype(np.float32)*.5-1
        np.testing.assert_array_equal(out.float().numpy(),expected[ids.numpy()])
        np.testing.assert_allclose(qt.float().numpy(),q,rtol=0,atol=.01)
        np.testing.assert_allclose(kt.float().numpy(),k,rtol=0,atol=.01)
        x=torch.ones((1,inp),dtype=torch.bfloat16)
        y=wkv(x).float().detach().numpy()
        w_expected=wq.astype(np.float32)*.25-.5
        np.testing.assert_allclose(y,np.ones((1,inp),np.float32)@w_expected.T,rtol=.01,atol=2)
        # Reader contract gate: parallel source-read order must not change output,
        # multiplicity, first-use LRU insertion order, repeat hits, or bounded eviction.
        from runtime.ds41.affine_safetensors import AffineRowLRU, SafeTensorMMap
        base='layers.1.engram.embed'
        contract_ids=np.asarray([[5,1,5],[0,4,1]],dtype=np.int64)
        eviction_ids=np.asarray([[2]],dtype=np.int64)
        old_workers=os.environ.get('DS41_ENGRAM_READ_WORKERS')
        old_min=os.environ.get('DS41_ENGRAM_PARALLEL_MIN_ROWS')
        try:
            os.environ['DS41_ENGRAM_READ_WORKERS']='1'
            os.environ['DS41_ENGRAM_PARALLEL_MIN_ROWS']='1'
            s1=SafeTensorMMap(root/'model-00047-of-00048.safetensors'); c1=AffineRowLRU(s1,base,max_rows=4)
            y1=c1.lookup(contract_ids); order1=list(c1._rows.keys()); y1_repeat=c1.lookup(contract_ids); e1=c1.lookup(eviction_ids)
            stats1=(c1.hits,c1.misses,c1.rows_read,list(c1._rows.keys()))
            os.environ['DS41_ENGRAM_READ_WORKERS']='4'
            s4=SafeTensorMMap(root/'model-00047-of-00048.safetensors'); c4=AffineRowLRU(s4,base,max_rows=4)
            y4=c4.lookup(contract_ids); order4=list(c4._rows.keys()); y4_repeat=c4.lookup(contract_ids); e4=c4.lookup(eviction_ids)
            stats4=(c4.hits,c4.misses,c4.rows_read,list(c4._rows.keys()))
            np.testing.assert_array_equal(y1,y4); np.testing.assert_array_equal(y1_repeat,y4_repeat); np.testing.assert_array_equal(e1,e4)
            assert order1==order4 and stats1==stats4, (order1,order4,stats1,stats4)
            assert len(c4._rows)<=4 and c4.parallel_batches>=1
            s1.close(); s4.close()
        finally:
            if old_workers is None: os.environ.pop('DS41_ENGRAM_READ_WORKERS',None)
            else: os.environ['DS41_ENGRAM_READ_WORKERS']=old_workers
            if old_min is None: os.environ.pop('DS41_ENGRAM_PARALLEL_MIN_ROWS',None)
            else: os.environ['DS41_ENGRAM_PARALLEL_MIN_ROWS']=old_min
        print({'status':'PASS','embed_shape':tuple(out.shape),'wkv_shape':tuple(wkv.weight.shape),'stats':emb.stats(),
               'parallel_contract':{'first_use_order':order4,'repeat_stats':stats4,'parallel_batches':c4.parallel_batches}})
        src.close();emb.source.close()

if __name__=='__main__':main()
