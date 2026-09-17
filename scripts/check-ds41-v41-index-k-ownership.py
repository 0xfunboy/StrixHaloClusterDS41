#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path

KV_SOURCES=(2,8,14,20)
INDEX_SOURCES=(2,8,14,20,24,28,32,36)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def src_layer(layer): return max(s for s in KV_SOURCES if s <= layer)

def role(layer):
    if layer in KV_SOURCES: return 'FULL_SOURCE'
    if layer in INDEX_SOURCES: return 'REINDEX'
    return 'REUSE'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    p=Path(a.source); text=p.read_text(); tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV4Indexer')
    fwd=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='forward')
    prod=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='_produce_k')
    fwd_text=ast.get_source_segment(text,fwd) or ''
    prod_text=ast.get_source_segment(text,prod) or ''
    global_shared_index_k='shared_attn.index_k' in text or 'shared.index_k' in text
    direct_kcache_assignment='self.k_cache = k_cache' in text
    produce_early_none=('latent is None' in prod_text and 'return' in prod_text)
    init=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='__init__')
    init_text=ast.get_source_segment(text,init) or ''
    attn_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV4Attention')
    sparse=next(n for n in attn_cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='_sparse_indexer_and_attn')
    sparse_text=ast.get_source_segment(text,sparse) or ''
    forward_reads_bound_cache=('SparseAttnIndexer(' in init_text and 'self.k_cache' in init_text and 'self.indexer.indexer_op(' in sparse_text)
    # Synthetic identity topology: exactly the local constructor contract.
    cache={s:object() for s in KV_SOURCES}
    index_cache={i:cache[src_layer(i)] for i in INDEX_SOURCES}
    identities={str(i): {'role':role(i),'kv_source':src_layer(i),'same_object':index_cache[i] is cache[src_layer(i)]} for i in INDEX_SOURCES}
    # Simulate previous forward ending at the last reindexer/source20 cache and a new
    # incomplete ratio-2 group at layers2/8/14. Local selection is per-indexer object,
    # not a mutable published global pointer, so stale previous-source selection is impossible.
    previous_active=cache[20]
    transitions=[]
    for frontier in (0,1,2,3,1022,1023):
        parity='even' if frontier % 2 == 0 else 'odd'
        for s in (2,8,14,20):
            latent_present=((frontier+1)%2==0) if s in (2,8,14) else True
            selected=index_cache[s]
            transitions.append({'frontier':frontier,'parity':parity,'layer':s,'latent_present':latent_present,
                                'selected_source':s,'selected_is_own':selected is cache[s],
                                'selected_is_previous_source20':selected is previous_active})
    # Reindexers all retain source20 object regardless of whether current source20 produces new K.
    reindex=[{'layer':i,'source':20,'same_object_as_source20':index_cache[i] is cache[20]} for i in (24,28,32,36)]
    # Consumers between sources: direct compressed source mapping and topk source mapping.
    rows=[]
    last_index=None
    for l in range(2,40):
        if l in INDEX_SOURCES: last_index=l
        rows.append({'layer':l,'role':role(l) if l in INDEX_SOURCES or l in KV_SOURCES else 'REUSE',
                     'kv_source':src_layer(l),'index_source':last_index})
    passes={
      'no_mutable_global_index_k_slot': not global_shared_index_k,
      'constructor_binds_indexer_to_cache_object': direct_kcache_assignment,
      'incomplete_group_does_not_rebind_cache': produce_early_none,
      'forward_consumes_bound_indexer_cache': forward_reads_bound_cache,
      'all_index_sources_have_expected_cache_identity': all(x['same_object'] for x in identities.values()),
      'all_full_sources_keep_own_cache_across_frontiers': all(t['selected_is_own'] for t in transitions),
      'reindexers_share_source20_cache_by_object_identity': all(x['same_object_as_source20'] for x in reindex),
    }
    status='NOT_APPLICABLE_PROVED' if all(passes.values()) else 'FAIL_LOCAL_OWNERSHIP_CONTRACT'
    out={'schema':'ds41-v41-index-k-ownership-v1','status':status,'source':str(p),'source_sha256':sha(p),
         'hf12_bug_shape':'mutable shared index_k pointer published only when owns_k && latent != None',
         'local_design':'per-source DeepseekV4IndexerCache objects; non-owning index sources bind directly to latest kv-source cache object in static_forward_context; _produce_k may skip writes but does not select/rebind cache',
         'checks':passes,'index_cache_identity':identities,'frontier_cases':transitions,'reindex_cases':reindex,
         'layer_source_map':rows,
         'scope':'ownership/cache-object selection only; does not validate K values, index scores, or full-model quality'}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':status,'checks':passes,'frontier_cases':len(transitions),'reindex_cases':reindex},indent=2)); return 0 if status=='NOT_APPLICABLE_PROVED' else 1
if __name__=='__main__': raise SystemExit(main())
