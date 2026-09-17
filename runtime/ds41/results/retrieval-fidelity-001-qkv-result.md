# RETRIEVAL FIDELITY 001 — Q/KV projection reference result

**Q/KV projection reference: PASS. Boundary e righe verificati:** layer2 `attn_norm.x` -> normalized KV endpoint on SWA rows895..1022 and1460..1587, plus final Q at positions1022/1587 after rank-local TP2 Q-b and compressed YaRN RoPE, on both ranks. **Discordanza localizzata: nessuna entro il gate.**

Frozen BF16 gate: rtol0.016/atol1e-5 + finite. Q: 0/65536 outside. KV: 0/262144 outside. Worst Q max-abs 0.0078125; worst KV max-abs 0.0078125. rel-L2/max-abs/ULP remain diagnostics only.

This certifies only the saved layer2 projection endpoints for code2k1588 request0. It does not certify `attn_norm.x` semantic correctness, later layers, or full retrieval. No runtime operator was reused by the reference; no inference/load/GPU replay/service mutation occurred.

Next discriminator is **not started**. Hypothesis: if the failure predates Q/KV, independently replay the already-saved layer2 entry -> MHC-pre -> attn_norm segment with capture-source weights. PASS moves the unverified boundary upstream; FAIL localizes only that segment after contract verification.
