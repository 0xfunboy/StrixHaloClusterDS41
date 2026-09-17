# RETRIEVAL FIDELITY 001 — Q/KV projection reference preregister

Frozen before fixture execution. Source is capture release `d4548014...`; packets are code2k1588 request0. Required layer2 attention tensors are BF16. Independent CPU reference uses direct GGUF BF16 bytes, FP32 accumulation, BF16 casts at runtime boundaries, explicit RMSNorm and pure V4.1 compressed YaRN RoPE. Q uses final row; KV uses only the already-associated128 SWA rows per chunk. TP2 Q-b rows are rank-local.

Gate fixed before results: finite and BF16 `rtol=0.016, atol=1e-5` for q_final and normalized KV. rel-L2/max-abs/ULP are diagnostics only. PASS is component-only; FAIL requires mapping/contract recheck before any patch.
