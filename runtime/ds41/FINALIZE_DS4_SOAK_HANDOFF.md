# FINALIZE DS4 SOAK — live handoff

Updated: 2026-09-17 during compact quality panel.
Phase: quality panel; Go coding IN_FLIGHT; soak BLOCKED.

## Live candidate
- Source/release: `9c13117f56fd81d03c8c610a5fb0148ccdc603b9` / `k2-mmq-engram-9c13117`.
- Epoch: `1789639579678586954`; K2/M4, canonical=1, MMQ=1, CED=0, MADV_RANDOM=1, Engram workers4/min_rows256.
- MMQ same-source gain: 3.149363x prefill throughput vs MMQ OFF.
- Engram same-source characterized gain: serial29.238172s vs parallel-confirm23.792395s (-18.626% time); two workers4 median22.320285s.
- CED remains `BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT`; do not reopen.

## Quality registry
- Preregister: `reports/DS41-Q2-001/final-quality-9c13117/preregister.json` SHA256 `31763c723908049e2c6b5373ad042a6e20700938efb7f0a19a08a4bdd6d49389`.
- Registry: `reports/DS41-Q2-001/final-quality-9c13117/registry.json`.
- code2k: current sample REUSED / semantic FAIL; not resent.
- tail1546: COMPLETE/FAIL, expected end5 actual end23; 1546/cache0; prefill20.797611s /74.335460TPS.
- docs2k: COMPLETE/FAIL, expected frozen facts; actual result58 / wrong middle+last; 1265/cache0; prefill20.476592s /61.777859TPS.
- JSON-none PASS; fresh STABLE-01 PASS; multi-turn ORBIT-17 PASS.
- C `c-frame-stream`: FAILED `INCOMPLETE_NO_FINAL`; 4096 completion all reasoning, zero final; tests NOT_EXECUTED => NON_VERIFIED; do not retry.
- Go `go-session-state`: IN_FLIGHT task `402e61b8a9701d2c65c3a244824e40e9`; reconcile, do not resubmit.
- Common-prefix logits cap1/cap32 and speed128 remain NOT_SENT.

## Decision / next exact act
1. Reconcile Go task to terminal; no concurrent model request.
2. Run exactly preregistered common-prefix logits cap1/cap32 and compare first-token/top2.
3. Run one contemporary frozen speed128 sample for decode non-regression.
4. Close quality verdict and update PLAN §2/§17/§36 plus this handoff.
5. Because code2k/tail/docs are already FAIL, soak is blocked unless the mandated discriminator demonstrates a runtime defect that is causally fixed and affected cases pass; do not reopen generic decode determinism.
