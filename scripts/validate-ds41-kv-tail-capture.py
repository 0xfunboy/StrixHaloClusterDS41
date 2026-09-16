#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def validate_one(d):
    assert d['schema']=='ds41-v2-kv-metadata-capture-v1'
    assert d['prompt_tokens']==1546
    ch=d['chunks']; assert len(ch)==2, len(ch)
    exp=[(0,1024,1024,[0,1023]),(1024,522,1546,[1024,1545])]
    for c,(before,scheduled,after,pr) in zip(ch,exp,strict=True):
        assert c['computed_before']==before
        assert c['scheduled']==scheduled
        assert c['computed_after']==after
        assert c['seq_len']==after
        assert c['seq_len_cpu_upper_bound']==after
        assert c['position_range']==pr
        assert c['positions']==list(range(pr[0],pr[1]+1))
        assert len(c['positions'])==scheduled
        for g in c['groups']:
            if g['mapping_enabled']:
                assert g['slot_consistency'] is True and g['first_bad'] is None
                assert len(g['slots'])==scheduled
                k=g['kernel_block_size']; need=pr[1]//k+1
                assert len(g['block_ids'])>=need
    # Logical slot visibility: every prompt position has one distinct enabled slot
    # per cache group across the two chunks. Physical ids themselves are rank-local.
    groups=len(ch[0]['groups'])
    for gi in range(groups):
        assert ch[1]['groups'][gi]['kernel_block_size']==ch[0]['groups'][gi]['kernel_block_size']
        if ch[0]['groups'][gi]['mapping_enabled']:
            slots=ch[0]['groups'][gi]['slots']+ch[1]['groups'][gi]['slots']
            assert len(slots)==1546 and len(set(slots))==1546
    return {'rank':d['rank'],'req_id':d['req_id'],'chunks':[(x['computed_before'],x['scheduled'],x['computed_after'],x['position_range']) for x in ch], 'groups':groups}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank0',required=True); ap.add_argument('--rank1',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    d0,d1=load(a.rank0),load(a.rank1); r0,r1=validate_one(d0),validate_one(d1)
    assert r0['chunks']==r1['chunks']
    # Do not compare physical block ids/slots across ranks.
    assert r0['groups']==r1['groups']
    for c0,c1 in zip(d0['chunks'],d1['chunks'],strict=True):
        for g0,g1 in zip(c0['groups'],c1['groups'],strict=True):
            assert g0['block_size']==g1['block_size'] and g0['kernel_block_size']==g1['kernel_block_size'] and g0['mapping_enabled']==g1['mapping_enabled']
    out={'schema':'ds41-v2-kv-tail-validation-v1','status':'PASS','prompt_tokens':1546,'rank0':r0,'rank1':r1,'interpretation':'Observed V2 metadata covers positions 0..1545 as 1024+522 with internally consistent block/slot mappings on both ranks. This does not prove attention numerical correctness.'}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
