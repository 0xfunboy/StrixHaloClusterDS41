# DS41 RECOVERY R1a — index-K ownership

Status: **NOT_APPLICABLE_PROVED**.

per-source DeepseekV4IndexerCache objects; non-owning index sources bind directly to latest kv-source cache object in static_forward_context; _produce_k may skip writes but does not select/rebind cache

Checks:
- no_mutable_global_index_k_slot: PASS
- constructor_binds_indexer_to_cache_object: PASS
- incomplete_group_does_not_rebind_cache: PASS
- forward_consumes_bound_indexer_cache: PASS
- all_index_sources_have_expected_cache_identity: PASS
- all_full_sources_keep_own_cache_across_frontiers: PASS
- reindexers_share_source20_cache_by_object_identity: PASS

Conclusion: HF discussion #12 is not directly applicable to the local vLLM cache-selection topology. The upstream bug depends on a mutable shared `index_k` selector; the local pin binds each index source to a cache object at construction time. Incomplete compression can skip writes, but cannot leave the indexer pointing at a later source cache.

Scope: ownership/cache-object selection only; K values, index scores and full-model quality remain separate.
