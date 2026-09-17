# DS41 RECOVERY UPSTREAM — handoff

## Authority
- Mandate: `DS41_RECOVERY_UPSTREAM_END_TO_END_2026-09-17.md`, read in full 2026-09-17.
- Truth: `/home/funboy/STRIX_CLUSTER_ACCELERATION_PLAN.md`.
- `FINALIZE DS4 SOAK` remains terminal. Q/KV `415507b` remains closed and is not replayed.

## Live state / R0
- Branch `exp/ds41-q2-001`.
- Serving retained untouched: `k2-prefill-5bdfed6`, source `5bdfed698ab97b6230db3fe8a7e37ff8e9b3d513`, epoch `1789651304744356945`, READY/idle at recovery start.
- Capture/reference source for offline R1: `k2-layer2diag-d454801`, source `d4548014d9903064074c2122bc92b690d366a3b6`.
- Preserve accelerated `9c13117f...` MMQ+Engram experimental release; no deletion or rewrite.

## R1a — upstream index-K ownership
- Upstream HF discussion #12 / commit `f5c8883826464ec59d2483e8cb18ae580123f5e9` fixes a mutable `shared_attn.index_k` selector: owner cache must be selected even when `latent=None`.
- Local pin has no mutable shared index-K selector. KV-source indexers 2/8/14/20 own distinct `DeepseekV4IndexerCache` objects; reindexers 24/28/32/36 bind directly to source20's cache object at construction through `static_forward_context`.
- Local `_produce_k` may skip K writes on incomplete groups but does not rebind/select another cache. Synthetic source-aware cases cover even/odd/incomplete frontiers and cross-forward stale-source20 scenario.
- Result: `NOT_APPLICABLE_PROVED` for HF#12 bug shape; this does not validate K contents/index scores.
- Evidence: `runtime/ds41/results/recovery-r1a-ownership.{json,md}`.

## Current phase
- R1b offline endpoints, no model load/service mutation:
  1. layer2 entry -> delayed/single-pass mHC-pre -> attn_norm;
  2. saved sparse-attention output -> inverse RoPE -> grouped WO_A -> WO_B TP2 reduction.
- Then R1c limited semantic-contract table (mHC/QAT/cache/Engram), followed automatically by R2 or fix/R3 per mandate.
## R1b — offline endpoints terminal
- PASS delayed/single-pass layer2 mHC-pre + attn_norm on both ranks/chunks. Previous-post residual and collapse are bit-exact; coefficient gates pass; attn_norm worst 1 BF16 ULP.
- PASS saved sparse-output -> inverse RoPE -> grouped WO_A -> WO_B -> TP2 endpoint. Worst rel-L2 `7.43e-4`, max-abs `0.0078125`; rank targets exact.
- Evidence: `runtime/ds41/results/recovery-r1b-result.{json,md}`. No model load/capture/service mutation.
- NEXT: finish R1c limited official/local semantic-contract table. If no causal defect is proved, enter bounded R2 automatically.
## R1c — semantic contracts terminal
- `MULTIPLE_CONTRACT_DIFFERENCES_NO_CAUSAL_DEFECT_PROVED`.
- Aligned/tested: delayed mHC/order, critical F32 params; HF#12 ownership not applicable locally.
- Non-equivalent: local window/compressed KV `fp8_ds_mla` vs V4.1 QAT reference; local gfx1151 index Q/K FP8 vs V4.1 FP4 QAT contract.
- Strong model-artifact difference: Engram sidecar is `Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP@802f1a0...` affine2 (embedding/WKV), not native FP8 Engram. TP split/parallel reader preserve this local source only.
- No causal patch from R1c. NEXT R2 same-input window-QAT differential offline on saved code2k1588; no model load yet.
## R2 — window-QAT same-input differential
- `MATERIAL_COMPONENT_EFFECT`: replacing only SWA128 rows with V4.1 window-QAT emulation changes rank1 chunk1 attention beyond frozen0.02/0.02 gate (3/16384, max-abs0.0703125). Other rank/chunks remain within gate.
- This proves a representation choice can materially change the saved component output, not which representation is end-to-end correct.
- NEXT: one bounded full-model localization window with an isolated window-QAT semantic QDQ patch, only if it can be inserted without changing unrelated contracts.
## R2 decision
- Window-QAT same-input effect is material, but exact semantic patch is `BLOCKED_BY_LOCAL_CACHE_ABI`: current decomposed path still writes hybrid `fp8_ds_mla`; exact full512/block32 QAT would require page/writer/gather changes and would overlap other non-equivalent contracts.
- No speculative local patch and no full-model localization window consumed.
- NEXT R4: isolated `kyuz0/ds4@7d0454b...` V4.1/gfx1151 source/build/help/ABI/resource gates only. No weights/download/service lifecycle until gates pass.
