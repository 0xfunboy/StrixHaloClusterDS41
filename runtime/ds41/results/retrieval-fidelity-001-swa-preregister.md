# RETRIEVAL FIDELITY 001 — SWA recent-KV preregister

Frozen before execution. Fixtures are **code2k1588 request0**, not discriminator1571. Logical association must pass first: chunk0 SWA ->895..1022, chunk1 ->1460..1587 on both ranks.

`kv_current_chunk` is post-normalization but pre-cache-insert/RoPE/quant; SWA `context_rows` is post-gather/dequant. Reference is vLLM `0bfb653d3b5161660db9ada0d84c2cdd60961de7` fused-insert math. Apply compressed V4.1 YaRN RoPE once to last64. Frozen gates: NoPE448 per-token max-abs <=16×max UE8M0 scale; RoPE64 <=1 BF16 ULP. Per-block values are diagnostics, not a new gate.

PASS is component-only. FAIL is initially a comparison discordance and must survive association/reference verification before any patch. CPU-only; no inference/GPU/service change.
