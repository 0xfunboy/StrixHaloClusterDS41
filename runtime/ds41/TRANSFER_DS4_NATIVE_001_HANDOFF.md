# TRANSFER DS4 → NATIVE 001 — handoff

## Phase
L0 COMPLETE / L1 PREPARED, before native model load.

## Reference/fallback
DS4 DOCUMENT PROFILE 002 terminal PASS: 24/24 soak, 8185.734s, finalizer QUALIFIED_LEFT_READY. DS4 is currently the qualified resident reference; K2 old rollback is not the automatic fallback.

## L0 result
- isolated repo worktree: `/home/funboy/worktrees/ds41-transfer-ds4-native-001`, branch `exp/ds41-transfer-ds4-native-001`;
- candidate vendor starts from exact existing 25-file MMQ/Engram/K2 overlay;
- opt-in `ds4-low-v1` modifies only V4.1 tokenizer/encoding;
- actual DeepseekV4Renderer full-ID equality vs DS4: 7/7 PASS;
- historical native profiles unchanged: 5/5 PASS;
- fail-closed invalid combinations: 6/6 PASS;
- target/DSpark single canonical prompt source contract PASS;
- patch fresh-apply: 27/27 byte-identical.

## L1 frozen candidate
DenseFix + Engram2, TP2/EP2, MMQ prefill ON, Engram workers4/min_rows256, canonical-prefill ON, K2=2, CED OFF, profile ds4-low-v1, cap2048, temp0/seed1, prefix cache OFF. Max6 document requests.

## NEXT
Commit/push L0. Prepare isolated native release on both nodes and run all model-free release/path/ABI checks. Then stop DS4 through its owner controller and execute one L1 native load + max6 runner. L1 PASS → L3; L1 FAIL → L2.
