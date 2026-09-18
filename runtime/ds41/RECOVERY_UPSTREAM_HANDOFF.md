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

## Acquisition reconcile (2026-09-18)
- NODE02 Q2 Internet acquisition is already COMPLETE and full SHA verified: 365713686528 B / 1ce6a8f8....
- Addendum pin d12db970... and acquired pin dd8a266f... resolve to the same HF file oid/LFS oid/size; no re-download.
- NODE01 mirror is BLOCKED_MARGIN: 372.8501 GiB available before mirror, 32.2527 GiB projected after, deficit 17.7473 GiB versus required 50 GiB. No alternate large filesystem.
- No deletion/mirror performed; K2 remains READY/idle. Continue only model-free recovery preparation until space reclamation is authorized.
- DS4 lifecycle preflight now passes binary identity, USB4, port-bind, K2 ownership and NODE02 asset checks, then fail-closes as BLOCKED_MODEL_MIRROR_NODE01 before any stop/start.

## Owner-authorized model cleanup (2026-09-18)
- Cleanup batch2 removed rejected/orphan Qwen W4A16, Qwen3.5 MTP, Flash-Next Q5/Q4 and Coder-Next FP8 as explicitly authorized; freed 334.158 GiB NODE01 / 204.914 GiB NODE02.
- Antirez Q2 mirror is COMPLETE + full SHA verified on NODE01; exact-size copy remains VERIFIED on NODE02.
- Removed on BOTH nodes: qwen3.8-27b-q5-xl, qwen3.5-9b-defiant-fable-bf16, qwen3.8-27b-fp8, qwen3-coder-next-80b-a3b-ud-q6-k, and historical broken deepseek-v4.1-flash-mixedq2 original.
- Freed 286.121 GiB per node. DenseFix/Engram/DSpark preserved and K2 remained READY.
- NODE01 now satisfies the >=50 GiB post-mirror margin; low-priority resumable USB4 mirror job `ds41-r4-q2-mirror.service` is IN_FLIGHT. Suspend it before measured/model-load windows.

## R4 DS4 quality terminal (2026-09-18)
- DS4 target-only 50/50 TCP recovered the non-ambiguous retrieval defect: tail1546 PASS, discriminator1571 exact PASS, four frozen independent retrieval holdouts 4/4 PASS.
- Frozen code2k/docs2k remain FAIL only on `middle_file`; both have even FILE-section counts (6/4), so the previously documented median-by-order ambiguity is isolated without changing expected values. Arithmetic result52 and first/last are correct.
- Function JSON/fresh/multiturn 3/3 PASS. Go coding PASS under frozen `go test -race`; C coding INCOMPLETE_NO_FINAL after 8192 reasoning tokens / zero final content.
- Initial upstream default TP gate750ms failed at layer35 on a40MiB transfer. Preregistered source-supported `DS4_TP_GATE_TIMEOUT_MS=5000` transport-only recovery completed all remaining cases without further TP failure.
- Strict quality remains unqualified; no performance benchmark or promotion. Restore K2 READY. Evidence `runtime/ds41/results/recovery-r4-ds4-quality-result.{json,md}` and `recovery-upstream-final.md`.

## Terminal operational state (2026-09-18)
- DS4 R4 stopped after quality decision; no promotion/performance arm.
- K2 `k2-prefill-5bdfed6` restored READY on epoch `1789712415010337206`; rank0/rank1/paired/gateway HTTP200.
- Final none-mode smoke returns exact `323`/stop with0 reasoning tokens.
- Recovery `delivery_complete=true`, `quality_qualified_scope=false`, `performance_target_met=false`.

## NEXT
- Recovery terminal: restore and retain `k2-prefill-5bdfed6` READY. DS4 remains an unpromoted alternative with retrieval-recovery evidence; no speed run under this quality result.
- Start exactly one Internet acquisition on NODE02 with pinned HF revision; persistent registry `NOT_SENT→IN_FLIGHT→COMPLETE/FAILED` and full SHA verification.
- Only after COMPLETE: resumable USB4 mirror to NODE01, verify size/SHA there, then packaging/lifecycle preregistration before stopping K2.
