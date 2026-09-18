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
- Startup supervisor immediately initiated K2 rollback on epoch `1789719735837225261`; no second DS4 ON while rollback is STARTING.
- Fix is launcher-only: remove the five literal escapes; bash syntax + invalid-role no-load guard PASS and fixed launcher replicated byte-identically to NODE02 (SHA256 `496ff6fb...`).

## NEXT
1. Wait for K2 rollback epoch `1789719735837225261` to become READY; do not issue another K2 or DS4 start meanwhile.
2. Then whole-pair K2 OFF through controller; DS4 USABLE ON exactly once and poll readiness with fixed launcher.
3. Run code2k-v2, docs2k-v2, C-off in frozen order.
4. If code/docs PASS: diagnostic P1/P2/P3, docs confirmation, then 4K→8K→16K dependent characterization.
5. If chat/document candidate remains qualified/useful: protected gateway candidate service gates, then conditional soak.
6. Leave DS4 only on full chosen-perimeter delivery; otherwise rollback K2.
