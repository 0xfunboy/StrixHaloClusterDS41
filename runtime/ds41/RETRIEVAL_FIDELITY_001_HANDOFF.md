# DS41 RETRIEVAL FIDELITY 001 — live handoff

Phase opened after terminal `FINALIZE DS4 SOAK` commit `2c44ae9c255fff5531fea1fbf80ea669ecc6ad1c`.

## Preserved results
- MMQ/Engram remain frozen experimental performance artifacts; no retuning.
- CED remains blocked; soak remains terminal/not reopened.
- Operational release before this control: `k2-prefill-5bdfed6` / `5bdfed698...`, K2 READY.

## Input/test closure
- Frozen discriminator: `reports/DS41-Q2-001/prefill-priority-001/code2k-extract.txt`, SHA `f5543bba60bbf849bb106bf9efa9ca5381b88a4c4c9281cf60de37a767258745`.
- Actual text contains begin17, middle23, end5; end is immediately before the final extraction instruction.
- Historical server accounting is1571 computed/cache0 on candidate and rollback.
- The discriminator asks direct extraction only; the separate `median-by-order` ambiguity of code2k is irrelevant here.
- Prior exact-prompt run also recovered the complete six-file list but returned end29, so missing-file output is not required for the end failure.

## Frozen discriminant
Run exactly one target-only M1 API control from the SAME release/source `5bdfed6`, leaving speculative config absent and retaining target switches/weights/tokenizer/TP2/EP2/Engram advice. Same prompt, reasoning none, temp0, seed1, max128, context65536.

- M1 PASS => K2/spec-verifier implicated.
- M1 same FAIL => K2 is not a necessary cause; narrow to target model/runtime/weights/reference contract.
- M1 different FAIL => K2 not sole explanation; compare signature before another control.

Preregister: `reports/DS41-Q2-001/retrieval-fidelity-001/preregister.json`.
No other generation request is authorized in this checkpoint.
