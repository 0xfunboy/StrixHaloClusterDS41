# TRANSFER DS4 → NATIVE 001 — L2 woa1 preflight

**PASS on NODE01 and NODE02.** No model load and no L2 quality request occurred during this preflight.

Release:
`/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001-woa1`

Code provenance: `9fe0496398aae695c9d2486ad7b234a8ee0fde80`. Attempt: `transfer-ds4-native-001-l2-m1-woa1`.

The release preserves the anon1 loader that completed the 77.14 GiB/rank Antirez load and adds exactly one compatibility branch for the ROCm WO_A direct-weight fast path. When the GGUF module exposes a quantized `weight_type`, the helper derives and validates logical dimensions from `GGML_QUANT_SIZES`, reuses `vllm_gguf_plugin.ops.ggml_dequantize` into BF16 once, reshapes and caches. Historical BF16/FP8 paths are unchanged.

Both nodes independently PASS:
- launcher syntax and py_compile;
- full L2 CPU contract: 1038 ordinary target + 8 native Engram = 1046/1046, anonymous staging, progressive clean-page eviction, native hash/token-map contract and real row264 decode;
- WO_A Q8_0 model-free gate: packed 4x34 -> logical 4x32, existing plugin dequantizer called once, cache reuse, bad geometry fail-closed, BF16 regression exact;
- fast Antirez identity against the existing both-node SHA receipt.

Cross-node hashes match:
- launch-node.sh: `507e1782a9cf6664520e36b48979cb47c8584e735e9173ad656fb70df4e99bc4`
- rocm_aiter_mla_sparse.py: `de03f58da44774faf7d216c0195d7a0d6db7488106f99bedd0d3e2d73d78b8a4`
- weight_utils.py: `5c991c5055bc1615745d4c644f93ce6d6ec69e4e22cb4c0d89f488e1ed727742`
- gguf_stream_cache.py: `6f28c0bb5d9794ed2051f6c1f35406820d4dd13617490d2012ed436cb82a3962`

Before switch: qualified DS4 READY/HTTP200, native OFF, K2 OFF.

**NEXT:** commit/push controller + preflight evidence, update PLAN/handoff, then one `woa1` M1 startup. No document request before rank0/rank1/paired HTTP200 and exact live release/attempt identity.
