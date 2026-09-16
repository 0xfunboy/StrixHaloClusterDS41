# DS41 M4 repeatability root cause

Status: **CAUSE_LOCALIZED_SPARSE_TOPK_ORDER**.

For chunk1 the repeated M4 requests have bit-exact upstream layer2 inputs, identical final-query Q, identical current KV projection, the same 512 compressed top-k indices as a set, identical SWA suffix, and identical 640 KV rows after logical-index alignment. Only the order of the compressed top-k differs. The captured sparse-attention outputs then differ by max-abs 0.001953125 on both ranks.

The isolated ROCm Triton sparse-attention replay reproduces each captured arm bit-exact. Canonical ordering of the same top-k set (SWA unchanged) makes the replay outputs bit-exact. This localizes the same-arm variability to non-canonical sparse-prefill top-k ordering feeding an order-sensitive BF16 reduction; it does not implicate BLOCK_M8.

Candidate fix: canonicalize valid prefill top-k indices immediately after indexer selection, only for metadata-declared prefill rows; keep invalid -1 entries at the end. Decode/verifier paths remain untouched.
