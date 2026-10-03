# DS4 V4.1 DSPARK 002 handoff

updated_at: 2026-09-21T20:27+02:00
phase: TERMINAL_P5
status: NOT_QUALIFIED
verdict: P2_CORRECTNESS_FAIL_AFTER_REFINEMENT_BUDGET
delivery_complete: false
owner: DS4_V41_DSPARK_002
epoch: 20260921T042636Z
raw: /home/funboy/reports/DS4-V41-DSPARK-002
worktree: /home/funboy/worktrees/ds4-v41-dspark-002
branch: exp/ds4-v41-dspark-002
head: 734f5a3b684a7deab05956cbaab41f9ff4c5d5a6

## Terminal facts

- P0 PASS; P1 COMPLETE/frozen.
- Refinement #1 `8850129`: fixed worker VERIFY ACK session identity; protocol first-fail resolved.
- capture-r1 36/36: pristine frontier127/B2 max_abs 1.4498415, top1 mismatch.
- Bounded numeric trace localized first divergence to layer0 Q-A projection/RMS path.
- Refinement #2 `734f5a3`: aligned verifier tiny-row Q8 projections with V4.1 M1 F32-input contract.
- capture-r2 36/36 still CORRECTNESS_FAIL: frontier127/B2 max_abs 1.60204029, B6 max_abs 1.77630997.
- refinement budget 2/2 exhausted. No third causal refinement.
- clean economic timing not run; do not label this result NEW_VERIFIER_NO_HEADROOM.
- P3/P4 skipped; sidecar never converted/loaded.
- No push/PR/promotion.

## Resident state

P5 restore PASS. E1 `ds4-speed-001-engram1` concurrent READY; VERIFY2/timing/DSpark OFF; gateway18224/tokenizer18223 HTTP200; final smoke `DSPARK002-P5-FINAL-OK` natural stop. No experimental unit remains.

Candidate release hashes match NODE01/NODE02:
- ds4 4ba1df2e68b48083efc0245cff897f437ef59c65ec2d41896f1967dcc8052bb5
- ds4-server d690b1f78e04788a16523265f65677e7f44eafe3af40b26163e2879d94cc8bc5
- ds4-bench d513633134ba159247c0e295d8f7734a176756ecf342442648c7828fe7d4469a

## NEXT EXACT ACTION

None inside this mandate. Preserve E1 and the raw evidence. Any attempt beyond `734f5a3` requires a new explicitly authorized campaign/refinement budget; do not rerun capture-r2 unchanged.
