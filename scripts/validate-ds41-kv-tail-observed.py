#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def one(d):
    assert d['schema']=='ds41-v2-kv-metadata-capture-v1' and d['prompt_tokens']==1546
    ch=d['chunks']; assert len(ch)>=1
    cursor=0; group_count=len(ch[0]['groups']); enabled=[g['mapping_enabled'] for g in ch[0]['groups']]
    all_slots=[[] for _ in range(group_count)]
    prev_blocks=[None]*group_count
    chunks=[]
    for c in ch:
        assert c['computed_before']==cursor
        assert c['computed_after']==c['computed_before']+c['scheduled']
        assert c['seq_len']==c['computed_after']==c['seq_len_cpu_upper_bound']
        exp=list(range(c['computed_before'],c['computed_after']))
        assert c['positions']==exp and c['position_range']==[exp[0],exp[-1]]
        assert len(c['groups'])==group_count
        for gi,g in enumerate(c['groups']):
            assert g['mapping_enabled']==enabled[gi]
            assert g['slot_consistency'] and g['first_bad'] is None
            assert len(g['slots'])==c['scheduled']
            if g['mapping_enabled']:
                if prev_blocks[gi] is not None:
                    assert g['block_ids'][:len(prev_blocks[gi])] == prev_blocks[gi]
                prev_blocks[gi]=g['block_ids']
                all_slots[gi].extend(g['slots'])
        chunks.append({'computed_before':c['computed_before'],'scheduled':c['scheduled'],'computed_after':c['computed_after'],'position_range':c['position_range']})
        cursor=c['computed_after']
    assert cursor==1546
    # All enabled groups except the observed sparse-SWA group retain globally unique
    # slots. Group2 legitimately recycles blocks older than its window. K2 has
    # sliding_window=128 and one extra retained token (num_spec_tokens=2).
    swa_group=2; swa_window=128; extra_retained=1
    for gi,on in enumerate(enabled):
        if not on:
            continue
        if gi != swa_group:
            assert len(all_slots[gi])==1546 and len(set(all_slots[gi]))==1546
            continue
        first_slots=ch[0]['groups'][gi]['slots']; second_slots=ch[1]['groups'][gi]['slots']
        old_pos={slot:pos for slot,pos in zip(first_slots,ch[0]['positions'],strict=True)}
        reused=[old_pos[slot] for slot in second_slots if slot in old_pos]
        skipped_before_second=max(0,ch[1]['computed_before']-swa_window+1-extra_retained)
        assert reused, 'expected SWA block reuse not observed'
        assert max(reused) < skipped_before_second, (max(reused),skipped_before_second)
    swa_reused_old_positions=[old_pos[slot] for slot in ch[1]['groups'][swa_group]['slots'] if slot in old_pos]
    return {'rank':d['rank'],'chunks':chunks,'group_count':group_count,'mapping_enabled':enabled,
            'groups':[{'group':i,'block_size':ch[0]['groups'][i]['block_size'],'kernel_block_size':ch[0]['groups'][i]['kernel_block_size'],'mapping_enabled':enabled[i]} for i in range(group_count)], 'swa_group':2, 'swa_window':128, 'swa_extra_retained':1, 'swa_reused_old_position_min':min(swa_reused_old_positions), 'swa_reused_old_position_max':max(swa_reused_old_positions), 'swa_skipped_before_second':skipped_before_second}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank0',required=True); ap.add_argument('--rank1',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    d0,d1=load(a.rank0),load(a.rank1); r0,r1=one(d0),one(d1)
    assert r0['chunks']==r1['chunks']; assert r0['groups']==r1['groups']
    nominal=[{'computed_before':0,'scheduled':1024,'computed_after':1024,'position_range':[0,1023]},{'computed_before':1024,'scheduled':522,'computed_after':1546,'position_range':[1024,1545]}]
    observed=r0['chunks']
    out={'schema':'ds41-v2-kv-tail-observed-validation-v1','status':'PASS','prompt_tokens':1546,
         'nominal_1024_522_match':observed==nominal,'observed_chunks':observed,'rank0':r0,'rank1':r1,
         'physical_block_ids_compared_across_ranks':False,
         'conclusion':'No violation in observed V2 metadata: contiguous positions 0..1545, computed/seq lengths reach1546, block-table prefixes persist across chunks, enabled-group slot mappings are internally consistent, and the only observed physical-slot reuse is group2 SWA using old positions256..767 that are below the computed skip threshold895. This does not prove attention numerical correctness.'}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
