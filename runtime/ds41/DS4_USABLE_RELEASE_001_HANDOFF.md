# DS4 USABLE RELEASE 001 — handoff

## Authority
- New phase after terminal Recovery ab41ca9; do not reopen K2 forensics.
- Truth: /home/funboy/STRIX_CLUSTER_ACCELERATION_PLAN.md.

## Entry state
- K2 5bdfed6 READY, epoch 1789712415010337206.
- DS4 units OFF; no stale DS4 job.
- Antirez Q2 full-size receipt+SHA already VERIFIED on both nodes; no new rehash/download.

## Frozen preload gates
- DS4 source/build 7d0454b4...; ctx16384; TCP/USB4; gate timeout5000ms; target-only/no DSpark.
- code2k-v2: N6 lower-middle section3 _ds41_artifact.py, expected unchanged, prompt1635 tokens.
- docs2k-v2: N4 lower-middle section2 pull_request_template.md, expected unchanged, prompt1312 tokens.
- dependent code4k/8k/16k v2 frozen at3492/7323/14877 prompt tokens.
- C-off v2: same fixture, prompt603 tokens, thinking=false + effort=none, cap8192, <=1 justified repair.
- Request collector simulated SSE/timeout/cancel PASS.
- CPU tokenizer sidecar reproduces pinned DS4 token IDs exactly on all six frozen v2 prompts.
- C fixture buggy FAIL / golden PASS.
- Frozen quality/performance runners are published before the next load. Perf independence gate: cached_tokens<=32; larger hits remain recorded and do not enter the independent median.

## Startup negative 001
- First USABLE ON attempt failed before any DS4 model load/request: generated launcher retained literal backslashes before five shell parameter expansions; shell reported `worker}: command not found` at line3.
- Startup supervisor rollback PASS: K2 epoch 1789719735837225261 returned READY with rank0/rank1/paired HTTP200.
- Fix is launcher-only: remove the five literal escapes; bash syntax + invalid-role no-load guard PASS and fixed launcher replicated byte-identically to NODE02 (SHA256 `496ff6fb...`).

## Startup negative 002
- Second ON reached coordinator launch but NODE02 worker never exec'd: systemd failed at STDOUT setup (status209) because the raw/log parent directory did not exist on NODE02.
- Coordinator was killed by supervisor after ~0.4s; pair never READY and no request was sent.
- Fix is controller-only: create RAW directory on both nodes before systemd-run. Automatic K2 rollback PASS: epoch 1789720141923773729 READY with rank0/rank1/paired HTTP200.

## Startup READY
- Third startup, with both setup fixes, is READY: coordinator+worker active, API200, 50/50 TCP/USB4, gate5000, ctx16384, target-only/noDSpark, Engram disk-only, planned memory82.67GiB/rank.
- K2 is OFF at startup-quality checkpoint.

## Quality terminal
- code2k lower-middle v2 FAIL: result52 and first correct, but middle=prompt.go and last=test-ds41-prefill-metrics.py; frozen expected unchanged.
- docs2k lower-middle v2 FAIL: result52 + first/last correct, middle=README.md instead of lower-center pull_request_template.md.
- C thinking-off PASS first attempt, zero repair: thinking=false + effort=none, 0 reasoning chars, natural stop; sanitizer/private tests PASS.
- Live prompt counts1629/1306/597 are exactly6 below preload CLI estimates1635/1312/603; preserve this audit mismatch and do not use CLI-matched tokenizer sidecar as product equivalence evidence.
- Document gate FAIL: performance P1/P2/P3, context4K→16K, product gateway and soak NOT_ADMITTED. C-off qualification remains specific and does not rewrite old C-low INCOMPLETE.

## NEXT
1. Stop DS4 whole pair and restore K2 5bdfed6 via controller.
2. Require K2 READY rank0/rank1/paired HTTP200 and final none-mode 323 smoke.
3. Persist terminal delivery: quality_qualified_scope=false, coding_c_thinking_off_qualified=true, performance_target_met=false, DS4_USABLE_RELEASE_001 NO_PROMOTION.
