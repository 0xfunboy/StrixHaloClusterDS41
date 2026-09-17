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
