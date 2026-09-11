#!/usr/bin/env python3
from __future__ import annotations
import json,struct,tempfile,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime.ds41.affine_safetensors import (
    AffineRowLRU, SafeTensorMMap, float32_to_bf16_words,
)


def pack2(q: np.ndarray) -> np.ndarray:
    q=np.asarray(q,dtype=np.uint32)
    assert q.shape[-1] % 16 == 0
    x=q.reshape(*q.shape[:-1],-1,16)
    shifts=np.arange(0,32,2,dtype=np.uint32)
    return np.bitwise_or.reduce(x << shifts, axis=-1).astype('<u4')


def write_fixture(path: Path, q, scales, biases):
    tensors=[]; offset=0; header={}
    for name,dtype,arr in [
        ('layer.embed.weight','U32',pack2(q)),
        ('layer.embed.scales','BF16',float32_to_bf16_words(scales)),
        ('layer.embed.biases','BF16',float32_to_bf16_words(biases)),
    ]:
        raw=np.ascontiguousarray(arr).tobytes()
        header[name]={'dtype':dtype,'shape':list(arr.shape),'data_offsets':[offset,offset+len(raw)]}
        tensors.append(raw);offset+=len(raw)
    header['__metadata__']={'format':'mlx'}
    raw=json.dumps(header,separators=(',',':')).encode()
    raw += b' ' * ((8-len(raw)%8)%8)
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(raw)));f.write(raw)
        for blob in tensors:f.write(blob)


def main():
    rows,width=5,256
    q=(np.arange(rows*width,dtype=np.uint32).reshape(rows,width)%4)
    scales=np.array([[0.25,0.5,1,2],[1,1.5,2,2.5],[.125,.25,.5,1],[2,1,.5,.25],[.75,.5,.25,.125]],np.float32)
    biases=np.array([[-1,-2,-3,-4],[0,1,2,3],[1,1,1,1],[-2,-1,0,1],[.5,.25,0,-.25]],np.float32)
    # Expected values must use BF16-rounded scale/bias, exactly as the file does.
    s32=(float32_to_bf16_words(scales).astype(np.uint32)<<16).view(np.float32)
    b32=(float32_to_bf16_words(biases).astype(np.uint32)<<16).view(np.float32)
    expected=q.astype(np.float32)*np.repeat(s32,64,axis=-1)+np.repeat(b32,64,axis=-1)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'fixture.safetensors';write_fixture(p,q,scales,biases)
        st=SafeTensorMMap(p)
        got=st.affine2_rows('layer.embed',[4,0,2])
        np.testing.assert_array_equal(got,expected[[4,0,2]])
        cache=AffineRowLRU(st,'layer.embed',max_rows=4)
        a=cache.lookup(np.array([[0,2],[4,2]]))
        np.testing.assert_array_equal(a,expected[[[0,2],[4,2]]])
        b=cache.lookup(np.array([2,0]))
        np.testing.assert_array_equal(b,expected[[2,0]])
        assert cache.rows_read==3 and cache.hits==2
        print({'status':'PASS','rows_read':cache.rows_read,'hits':cache.hits,'misses':cache.misses,'shape':got.shape})
        st.close()

if __name__=='__main__':main()
