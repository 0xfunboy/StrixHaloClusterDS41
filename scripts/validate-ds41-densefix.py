#!/usr/bin/env python3
"""Independent validator for DS41 densefix artifacts.

Independence from the converter:
  * source E4M3 bytes are interpreted with PyTorch float8_e4m3fn;
  * E8M0 scales use an explicit exponent-bit upcast matching the V4.1 contract;
  * BF16 rounding is PyTorch's CPU cast, not the converter's NumPy RNE code.
The validator compares every repaired BF16 value and hashes every byte outside
patch intervals, then emits final repaired-shard SHA256 values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / ".vendor" / "llama-v41" / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))
from gguf import GGUFReader, GGMLQuantizationType

CHUNK = 64 * 1024 * 1024


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def compare_block_independent(weight_u8: np.ndarray, scale_u8: np.ndarray, actual_u16: np.ndarray) -> tuple[int,int]:
    # Copy ensures PyTorch owns writable CPU storage and no converter view is reused.
    w = torch.from_numpy(np.array(weight_u8, copy=True, dtype=np.uint8)).view(torch.float8_e4m3fn).float()
    sb = torch.from_numpy(np.array(scale_u8, copy=True, dtype=np.uint8)).to(torch.int32)
    s = torch.bitwise_left_shift(sb, 23).view(torch.float32)
    scale_vec = torch.repeat_interleave(s, 32)[: w.shape[1]]
    f = w * scale_vec.unsqueeze(0)
    if not bool(torch.isfinite(f).all()):
        raise RuntimeError("independent source dequant produced non-finite values")
    exp_bits = f.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    act_bits = np.asarray(actual_u16, dtype=np.uint16)
    neq = exp_bits != act_bits
    return int(neq.sum()), int(neq.size)


def update_unpatched_hash(h, chunk: bytes, base: int, intervals: list[tuple[int,int]], idx: int) -> int:
    end = base + len(chunk)
    while idx < len(intervals) and intervals[idx][1] <= base:
        idx += 1
    cursor = base
    j = idx
    while j < len(intervals) and intervals[j][0] < end:
        a,b = intervals[j]
        if cursor < min(a,end):
            h.update(chunk[cursor-base:min(a,end)-base])
        cursor = max(cursor, min(b,end))
        if b <= end: j += 1
        else: break
    if cursor < end:
        h.update(chunk[cursor-base:])
    return idx


def sha_payload(path: Path, offset: int, nbytes: int) -> str:
    h=hashlib.sha256(); left=nbytes
    with path.open('rb',buffering=0) as f:
        f.seek(offset)
        while left:
            b=f.read(min(8*1024*1024,left))
            if not b: raise RuntimeError(f"short payload read {path}")
            h.update(b); left-=len(b)
    return h.hexdigest()


def validate(args) -> None:
    manifest=json.loads(Path(args.manifest).read_text())
    cache=Path(args.cache); dest=Path(args.dest)
    precopy=json.loads((dest/'.densefix-precopy.json').read_text())
    repair=json.loads((dest/'.densefix-repair.json').read_text())
    result={
        'schema':'ds41-densefix-validation-v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'tool_commit':manifest['tool_commit'],'tensor_results':{},'shards':{},'summary':{}
    }

    # Structural GGUF re-open: selected names, offsets, BF16 type and sizes must be unchanged.
    structural={}
    for shard in sorted(manifest['gguf_shards']):
        r=GGUFReader(str(dest/shard),'r')
        by={t.name:t for t in r.tensors}
        structural[shard]=by
    total_bad=total_vals=0
    for i,e in enumerate(manifest['entries'],1):
        t=structural[e['gguf_shard']].get(e['gguf'])
        if t is None: raise RuntimeError(f"missing repaired tensor {e['gguf']}")
        if t.tensor_type != GGMLQuantizationType.BF16 or int(t.data_offset)!=int(e['gguf_offset']) or int(t.n_bytes)!=int(e['gguf_nbytes']):
            raise RuntimeError(f"GGUF structure changed for {e['gguf']}")
        m,k=map(int,e['source_shape']); sm,sk=map(int,e['scale_shape'])
        w=np.memmap(cache/e['cache_weight_file'],mode='r',dtype=np.uint8,shape=(m,k))
        s=np.memmap(cache/e['cache_scale_file'],mode='r',dtype=np.uint8,shape=(sm,sk))
        fd=os.open(dest/e['gguf_shard'],os.O_RDONLY)
        bad=vals=0
        try:
            for r0 in range(0,m,32):
                r1=min(m,r0+32)
                nrows=r1-r0
                blob=os.pread(fd,nrows*k*2,int(e['gguf_offset'])+r0*k*2)
                if len(blob)!=nrows*k*2: raise RuntimeError(f"short repaired read {e['gguf']}")
                actual=np.frombuffer(blob,dtype='<u2').reshape(nrows,k)
                b,n=compare_block_independent(np.asarray(w[r0:r1]),np.asarray(s[r0//32]),actual)
                bad+=b; vals+=n
        finally:
            os.close(fd)
        payload_sha=sha_payload(dest/e['gguf_shard'],int(e['gguf_offset']),int(e['gguf_nbytes']))
        rep=repair['completed'][str(e['index'])]
        if payload_sha != rep['repaired_payload_sha256']:
            raise RuntimeError(f"payload SHA changed after repair {e['gguf']}")
        result['tensor_results'][str(e['index'])]={
            'gguf':e['gguf'],'source':e['source'],'values':vals,'mismatch_values':bad,'payload_sha256':payload_sha,
            'finite':True,'status':'PASS' if bad==0 else 'FAIL'
        }
        total_bad+=bad; total_vals+=vals
        if bad: raise RuntimeError(f"independent BF16 mismatch {e['gguf']}: {bad}/{vals}")
        if i==1 or i%8==0 or i==len(manifest['entries']):
            print(f"VALIDATE_TENSOR_PROGRESS tensors={i}/{len(manifest['entries'])} values={total_vals} mismatches={total_bad}",flush=True)

    # Strong unrelated-byte check: every byte outside all patched payload intervals
    # must hash exactly as it did during the independent physical copy.
    final_sums=[]
    for shard in sorted(manifest['gguf_shards']):
        p=dest/shard
        intervals=sorted((int(x['start']),int(x['end'])) for x in manifest['gguf_shards'][shard]['patch_intervals'])
        hfull=hashlib.sha256(); hun=hashlib.sha256(); idx=0; pos=0
        with p.open('rb',buffering=0) as f:
            while True:
                b=f.read(CHUNK)
                if not b: break
                hfull.update(b); idx=update_unpatched_hash(hun,b,pos,intervals,idx); pos+=len(b)
        expected_un=precopy['shards'][shard]['unpatched_sha256']
        got_un=hun.hexdigest()
        if got_un!=expected_un:
            raise RuntimeError(f"non-repaired bytes changed in {shard}: {got_un} != {expected_un}")
        full=hfull.hexdigest(); final_sums.append(f"{full}  {shard}")
        result['shards'][shard]={
            'size':pos,'repaired_sha256':full,'unpatched_sha256':got_un,'unpatched_matches_original':True,
            'patch_count':len(intervals),'original_sha256':precopy['shards'][shard]['original_sha256']
        }
        print(f"VALIDATE_SHARD {shard} sha256={full} unpatched=PASS",flush=True)
    (dest/'SHA256SUMS.densefix').write_text('\n'.join(final_sums)+'\n')
    result['summary']={'status':'PASS','tensors':len(manifest['entries']),'values_checked':total_vals,'mismatch_values':total_bad,'unpatched_bytes':'BYTE_IDENTICAL_BY_STREAM_HASH'}
    atomic_json(dest/'densefix-validation.json',result)
    print(json.dumps(result['summary'],indent=2))


def selftest() -> None:
    w=np.array([[0x71]],dtype=np.uint8); s=np.array([0x73],dtype=np.uint8)
    actual=np.array([[0x3D10]],dtype=np.uint16)
    bad,n=compare_block_independent(w,s,actual)
    assert (bad,n)==(0,1)
    # Wrong historical artifact value must fail the independent path.
    bad2,n2=compare_block_independent(w,s,np.array([[0x2880]],dtype=np.uint16))
    assert (bad2,n2)==(1,1)
    print(json.dumps({'status':'PASS','correct_bits':'0x3d10','bad_historical_bits_rejected':'0x2880'}))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('selftest','validate'))
    p.add_argument('--manifest',default=str(ROOT/'reports/DS41-Q2-001/densefix/densefix-manifest.plan.json'))
    p.add_argument('--cache',default='/home/funboy/models/ds41/densefix-source-cache')
    p.add_argument('--dest',default='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix')
    args=p.parse_args()
    if args.command=='selftest': selftest()
    else: validate(args)

if __name__=='__main__': main()
