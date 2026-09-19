# TRANSFER DS4 → NATIVE 001 — L2 M1 startup negative 004

**Verdict:** anon1 solved the mmap/SVM memory blocker and completed model load on both ranks, but startup failed in the initial warmup/profile forward on one localized GGUF WO_A fast-path incompatibility. **0 L2 quality requests were sent.**

Startup identity:
- epoch `1789782496195400259`
- rank0 InvocationID `d8cad43f1ecb408eb7738eae608d9003`
- rank1 InvocationID `f9ac3bc8cf3f4ac0a2f3a497cd046fb6`
- release `native-antirez-m1-transfer001-anon1`
- attempt `transfer-ds4-native-001-l2-m1-anon1`

## Memory contract — PASS for this startup

Anonymous staging crossed the previous SVM failure point. Kernel SVM-failure count during anon1 startup was **0 on both nodes**.

Both ranks completed the Antirez GGUF load:
- rank0: `load_weights=217.229s`, total load `232.222s`
- rank1: `load_weights=203.232s`, total load `229.720s`
- vLLM reports **77.14 GiB model memory per rank**

The failure occurred only after load/process_weights, during the first warmup/profile forward.

## Terminal format failure

Both ranks fail identically in:
`rocm_inv_rope_einsum -> _get_cached_wo_a_bf16`

with:
`RuntimeError: shape '[4, 1024, 4096]' is invalid for input of size 17825792`

The fast path directly executes:
`wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)`

All 40 Antirez WO_A tensors have the same source contract:
- logical GGUF shape: `4096 x 8192`
- packed data shape: `8192 x 4352`
- GGML type: **Q8_0 (8)**
- bytes/tensor: `35,651,584`

The old DenseFix target has the **same logical shape** but BF16 storage. Under TP2 the Antirez packed Q8_0 parameter has 17,825,792 storage elements, while the logical rank tensor is 4×1024×4096 = 16,777,216. Q8_0 geometry is block32/type34, giving the correct logical rank shape 4096×4096 after dequantization.

The normal GGUF linear path already supports Q8_0 through `vllm_gguf_plugin.ops.ggml_dequantize`; static scoped audit found only this one direct-weight-view bypass in the V4.1 ROCm path.

## Recovery

Both failed DS41 units were already OFF-verified, the stale owner receipt was reconciled with `CLUSTER_OFF_RECONCILED`, and qualified DS4 DOCUMENT PROFILE 002 is restored **READY / HTTP200**. K2 remains OFF.

DS4 restore InvocationIDs:
- coordinator `1178a56893b44accabcb4c508062f5ef`
- worker `d8935559cfe9449d81a258fcdb861c74`

## NEXT

One scoped L2a compatibility fix only: when WO_A is GGUF-quantized, use its existing GGUF `weight_type`, derive/validate the logical dimensions from `GGML_QUANT_SIZES`, call the already-existing plugin `ggml_dequantize(..., BF16)` once, reshape and cache. Existing BF16/FP8 behavior remains unchanged.

No new kernel, weight conversion, quantization, driver or infra change. If another unrelated packed-format bypass appears after this single fix, close L2 rather than opening broad format forensics.
