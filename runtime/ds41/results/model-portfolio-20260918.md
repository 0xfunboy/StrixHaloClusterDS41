# Strix Halo model portfolio — 2026-09-18

This is a practical reuse ranking, not a universal intelligence leaderboard: workloads, quantizations and quality panels differ.

## Best retained / historically validated

1. **GLM-5.3-Flash CIRU W4/IU4 + DFlash2 k5** — best fully integrated general-purpose/product path. 24.851 decode TPS, 23.667 HTTP TPS; repeat/fidelity/API qualified in the recorded scope. Later harder panel: 7 PASS / 5 INCOMPLETE, not five wrong answers. Current retained path: `/home/funboy/models/ciru-glm53-flash`.
2. **Qwen3.8-Flash-Next IQ3_XXS original / EngramHalo** — best speed/intelligence/storage candidate. Target about 23.49 TPS; reasoning path median about 29.20 TPS, 8/8 new reasoning PASS; utility thinking-off 33.49–45.42 TPS with stress PASS. New xhigh coding panel ended 12/12 INCOMPLETE, so not universal coding qualification. Retained on NODE01 at `/home/funboy/models/gguf/qwen3.8-flash-next-unsloth-iq3-xxs` (~76.33 GiB).
3. **Qwen3.8-Flash-Next IQ4_XS uncensored + MTP** — strong fast local alternative. 34.771 TPS on two distinct code400 workloads, zero major faults in decision runs. Different weights/provenance from IQ3; retain when uncensored variant matters. Base retained on both nodes (~95.48 GiB each).
4. **Qwen3-Coder-Next 80B-A3B GGUF Q6** — historically excellent specialist: 48.7048 TPS decode / 810.996 prefill at ctx2048, 43.5576 decode at ctx32704. Removed on both nodes by owner on 2026-09-18 because it is no longer needed.
5. **Qwen3.8-27B dense Q5 + local MTP5** — historically fast but lower-capability dense model: 38.89698 TPS on the frozen short DS4 corpus, 52/52 there. Removed on both nodes by owner on 2026-09-18.

## Research / not currently quality-qualified

- **DeepSeek V4.1 DenseFix + DSpark K2** — strong architecture and current rollback serving path. Frozen K2 vs K1 panel: 19.4318 vs 16.8350 TPS with arithmetic/coding/JSON/reasoning-high PASS, but later code2k/tail/docs retrieval panel FAILs. Keep during Recovery; do not treat as final-quality qualified.
- **DeepSeek V4.1 Antirez Q2 on DS4** — current R4 alternative. Same exact Q2 now VERIFIED on NODE02 and being mirrored to NODE01; local quality/performance panel not run yet, so it is intentionally unranked until R4/R5.

## Lower-value but functional historical paths

- **Qwen3.8-Flash-Next Q5 original** — target-only ~18.109 TPS; functional/API history but superseded for practical speed by IQ3/IQ4.
- **Qwen3.8-Flash-Next Q4 original** — ~18.32 target / ~18.70 MTP3 in recorded runs; MTP qualification had repeat/reference problems. Still useful only as a higher-precision/original capacity reference.
- **Qwen3-Coder-Next FP8 vLLM TP2** — 19.264 TPS, 9/9 short requests, explicitly rejected versus local GGUF Q6 48.705 TPS.
- **Qwen3.8-27B Quark W4A16 vLLM TP2** — ~20.973 TPS at decision ISL2048, quick quality only, full quality blocked; explicitly rejected versus llama.cpp local.

## Cleanup candidates for owner review

High confidence:
- `/home/funboy/models/vllm/qwen3-coder-next-fp8` — **DELETED BOTH NODES 2026-09-18**; slower rejected duplicate path.
- `/home/funboy/models/vllm/qwen3.8-27b-amd-quark-awq-int4-w4a16` — **DELETED BOTH NODES 2026-09-18**; branch closed/rejected.
- `/home/funboy/models/gguf/qwen3.5-9b-defiant-fable-q5-mtp` — **DELETED BOTH NODES 2026-09-18**; orphan after BF16 base removal.

Likely removable if old Qwen references are no longer wanted:
- `/home/funboy/models/gguf/qwen3.8-flash-next-unsloth-q5-k-xl` — **DELETED 2026-09-18** (was NODE01 only); slower ~18.1 TPS original Q5 path.
- `/home/funboy/models/gguf/qwen3.8-flash-next-original-q4-k-xl` — **DELETED BOTH NODES 2026-09-18**; old original/capacity reference retired.

Conditional / after Recovery closes:
- NODE01 `/home/funboy/models/ds41/engram2-tp2/rank1` — ~28.70 GiB staging replica; NODE01 live rank uses rank0, NODE02 uses rank1.
- NODE01 `/home/funboy/models/ds41/engram2-source` — ~57.31 GiB source used to reconstruct partitions; not opened by serving, but useful for reproducibility until Recovery closes.
- NODE01 `/home/funboy/models/ds41/densefix-source-cache` — ~6.08 GiB repair source cache; safe candidate only after DenseFix recovery/audit no longer needs local reconstruction.

Keep:
- DenseFix, DSpark sidecar, active Engram rank partitions.
- GLM CIRU.
- Qwen Flash-Next IQ3 and IQ4 unless the owner explicitly chooses one checkpoint/provenance only.
