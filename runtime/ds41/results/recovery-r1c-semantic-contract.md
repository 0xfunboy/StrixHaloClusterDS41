# DS41 RECOVERY R1c — semantic contract table

Status: **MULTIPLE_CONTRACT_DIFFERENCES_NO_CAUSAL_DEFECT_PROVED**.

| Contract | Official/reference | Local capture pin | Status |
|---|---|---|---|
| mHC params/order | V4.1 single-pass/delayed mHC; critical coeff parameters FP32 | hc fn/scale/base F32; delayed carried-pre_mix path | `ALIGNED_TESTED_LAYER2` |
| attention sink/router bias precision | critical sink/bias parameters retained FP32 in reference | blk.2.attn_sinks F32; blk.2.exp_probs_b F32 | `ALIGNED_REPRESENTATION` |
| window KV QAT | post-norm+RoPE full window-KV quantized FP8 with block32 UE8M0/E8M0 scale contract | fp8_ds_mla hybrid: NoPE448 FP8 OCP with seven UE8M0 block64 scales; RoPE64 stored BF16 | `CONTRACT_DIFFERENCE_EFFECT_UNPROVEN` |
| compressed KV QAT | compressed latent uses V4.1 FP4 activation quantization with block16 and checkpoint/reference scale contract | compressed latent written through V4-style fp8_ds_mla hybrid: NoPE448 FP8 block64 UE8M0 + RoPE64 BF16 | `CONTRACT_DIFFERENCE_EFFECT_UNPROVEN` |
| indexer K/Q QAT | V4.1 index query/key contract uses FP4 block32 QAT/scale metadata in reference tests | gfx1151 path use_fp4_cache=False: K is per-token FP8 128-wide with one fp32 scale; Q is FP8 per token/head with scalar scale folded into index weights | `CONTRACT_DIFFERENCE_EFFECT_UNPROVEN` |
| index-K ownership/lifetime | HF discussion#12 fixes mutable shared index_k selector on incomplete compression step | distinct per-source DeepseekV4IndexerCache; reindexers directly bind source20 cache object; no mutable shared selector | `HF12_NOT_APPLICABLE_PROVED` |
| Engram tokenizer/hash | compressed-vocab mapping, rolling n-gram XOR multipliers, per-head prime buckets/offsets | vendored V4.1 hash/gating code retained; config compressed_vocab_size=99092, max_ngram4, heads8, layers1/14 | `ALIGNED_ALGORITHM_CONFIG_NOT_CROSS_ENGINE_PROVED` |
| Engram row/linear precision | native V4.1 Engram table path stores FP8 rows with UE8M0 per32 scales; reference checkpoint precision is part of model contract | Vontra MLX 2-bit MTP source: embedding and WKV affine2 U32 + BF16 scales/biases; q/k BF16; dequantized BF16 by DiskAffineEngram. TP2 partition preserves those source bytes/ranges but not native FP8 representation | `STRONG_CONTRACT_DIFFERENCE_CAUSALITY_UNPROVEN` |

**Decision:** R1 does not prove one causal runtime bug. HF#12 is excluded locally; mHC/output endpoints conform. Window/compressed/index QAT and Engram storage are non-equivalent model/runtime contracts, so R2 must measure a same-input effect before any local patch.

Engram local source: `Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP@802f1a00982705d81b79ad1c83aa0ccc0b863ebc`; this is not native FP8 Engram.
