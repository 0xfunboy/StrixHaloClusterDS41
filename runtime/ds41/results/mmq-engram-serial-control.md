# MMQ + Engram same-source serial reader control

- Same source/release `9c13117f...` / `k2-mmq-engram-9c13117`; only reader workers changed 4→1.
- code-2k 1588/cache0: prefill **29.238172s / 54.312561 tok/s**, TTFT 29.606769s, wall 31.974004s.
- Semantic FAIL: result47, first/last correct, median invalid.
- Pre-request whole-sidecar mincore: NODE01 ~1.18MB each, NODE02 ~1.38–1.72MB (<0.012%): nearly empty OS page cache. This is an observable page-cache state, not a physical SSD cold claim.
- Initial parallel arm was 20.848176s / 76.169734 tok/s; observed delta 8.389996s. One parallel confirmation is required because its pre-request OS residency was not recorded.
