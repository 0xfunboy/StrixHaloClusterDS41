# DS41 RECOVERY UPSTREAM — handoff

## Authority
- Mandate: `DS41_RECOVERY_UPSTREAM_END_TO_END_2026-09-17.md`.
- Truth: `/home/funboy/STRIX_CLUSTER_ACCELERATION_PLAN.md`.
- FINALIZE DS4 SOAK terminal; Q/KV `415507b` closed; preserve MMQ/Engram `9c13117f...` experimental release.

## Live service
- `k2-prefill-5bdfed6`, source `5bdfed698...`, epoch `1789651304744356945`, READY/idle at latest check.
- Recovery R0-R4 gates have not mutated serving.

## R1 terminal
- R1a HF#12 ownership: `NOT_APPLICABLE_PROVED`; local vLLM binds per-source cache objects, no mutable global `index_k` selector.
- R1b: delayed V4.1 mHC-pre/attn_norm PASS; sparse-output→inverse-RoPE→WO_A→WO_B/TP2 endpoint PASS.
- R1c: multiple contract differences, no single causal local bug. Strongest artifact difference: local Engram2 is Vontra MLX affine2, not native V4.1 FP8 Engram.

## R2 terminal
- Same-input window-QAT differential is material only on rank1/chunk1 (3/16384 beyond frozen attention gate, max-abs0.0703125).
- Exact local semantic patch blocked by current `fp8_ds_mla` cache ABI: preserving V4.1 full512/block32 QAT would also require page/writer/gather/backend changes. No speculative patch and no full-model localization window consumed.

## R4 current
- DS4 isolated pin: `kyuz0/ds4@7d0454b4e32ef1e90235f2b001d6643b5934438c`.
- Build PASS both nodes with bundled HIP7.15.26333, gfx1151; binaries bit-identical; help/eval/linking PASS.
- GPU-visible pool ~120 GiB/rank; expected cluster resident weight ~80.6 GiB/rank, Engram disk-backed.
- Q2 pinned: `antirez/deepseek-v4.1-flash-gguf@dd8a266f7145edc19e2334b46e19b6821f221dc7`, `365713686528` bytes, SHA256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.
- Disk gate PASS recorded: NODE02 post-download user margin ~353.0 GiB; NODE01 post-mirror user margin ~32.3 GiB (~125.4 GiB free incl. reserved). Single payload only; no historical weights deleted.
- USB4 TCP link: NODE01 `10.55.0.1`, NODE02 `10.55.0.2`.

## NEXT
- Start exactly one Internet acquisition on NODE02 with pinned HF revision; persistent registry `NOT_SENT→IN_FLIGHT→COMPLETE/FAILED` and full SHA verification.
- Only after COMPLETE: resumable USB4 mirror to NODE01, verify size/SHA there, then packaging/lifecycle preregistration before stopping K2.
