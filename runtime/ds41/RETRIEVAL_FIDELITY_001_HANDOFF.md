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

## M1 control terminal
- Epoch `1789650706117112870`, same `5bdfed6`, target-only M1; no speculative config.
- Direct paired request exactly once after a frontend-only HTTP503 setup-negative that did not reach the model.
- 1571 computed/cache0; prefill92.416969s /16.999043 tok/s; TTFT92.535083s; wall98.360220s.
- FAIL: begin17, middle23, **end41**; files [`__init__.py`,`api.py`,`artifact.json`,`prompt.go`,`test-ds41-prefill-metrics.py`,`envelope_test.go`].
- K2 rollback FAIL was end29 with a different list. Decision: `K2_NOT_NECESSARY_FOR_FAILURE`; signature is mode-sensitive, so target model/runtime/weights/reference remains open.
- NEXT: restore K2 `5bdfed6` READY, then inventory only already-available independent references; no new generation until a reference is preregistered.

## Sparse-attention independent reference frozen
- Existing full-model independent engine is unavailable without a new build/port, so none is introduced.
- Vendored vLLM provides `_ref_sparse_prefill_ragged` and upstream gate `atol=rtol=0.02`.
- Frozen control: saved real code2k request0, chunk0+1, both ranks; CPU-only reference vs captured ROCm kernel output.
- PASS excludes sparse-attention arithmetic at those packets only; FAIL localizes that component.
- Preregister: `runtime/ds41/results/retrieval-fidelity-001-sparse-reference-preregister.{json,md}`.

## Sparse-attention independent reference terminal
- PASS rank0+rank1, chunk0+chunk1: 65,536 elements, 0 outside upstream `atol=rtol=0.02`.
- Worst max-abs `0.0078125`; worst rel-L2 `0.00114327`. CPU-only, no model load/generation/GPU kernel.
- Decision: `SPARSE_ATTENTION_ARITHMETIC_EXCLUDED_AT_CAPTURED_LAYER2_PACKETS`.
- Scope limit: ordered Q/KV/sink arithmetic only; selected-context correctness/indexer remains open.
- K2 rollback restored and READY/idle epoch `1789651304744356945`.
- Provenance correction: these packets are code2k1588 request0, not discriminator1571; do not transfer token coordinates between them.
- NEXT: CPU-only SWA recent-KV reference on saved rank0+rank1 code2k1588 packets. Verify logical association and capture boundary before comparison; upstream gates fixed as NoPE448 <=16×per-token max UE8M0 scale and RoPE64 <=1 BF16 ULP. No inference/GPU replay/load/indexer capture.

## SWA recent-KV terminal
- PASS on saved code2k1588 request0, both ranks/chunks; this is not discriminator1571.
- Association PASS: chunk0 logical895..1022; chunk1 logical1460..1587; source rows are the verified chunk tails on both ranks.
- 512 rows checked. NoPE448:0 tokens outside frozen16×max-scale gate,0 diagnostic blocks outside own16×scale. RoPE64:0 tokens outside <=1ULP; worst1ULP.
- Same code2k1588 `DISTANT_FACT_END=5` localizes around1490..1498, inside verified final SWA1460..1587.
- Decision: `SWA_RECENT_KV_CONFORMS_AT_SAVED_CODE2K1588_LAYER2_PACKETS`; component-only, no overall retrieval claim.
- NEXT: no SWA patch and no automatic indexer capture. Any further discriminator must be derived from this PASS and use saved tensors first or be separately preregistered. K2 service remains READY/idle.

## Q/KV projection reference frozen
- NEXT from SWA PASS: CPU-only endpoint reference on saved code2k1588 request0, capture source `d4548014...`; no model request/load.
- Input `layer2_full[*].attn_norm.x`; KV endpoint `kv_current_chunk` after wkv+kv RMSNorm and before RoPE; Q endpoint `q_final` after wq_a+q RMSNorm+rank-local wq_b+RoPE.
- All required GGUF layer2 tensors are BF16. TP2 Q-b rows: rank0 0..16383, rank1 16384..32767.
- Frozen gate: finite + BF16 upstream defaults `rtol=0.016, atol=1e-5`; rel-L2/max-abs/ULP diagnostic only.
- Preregister `runtime/ds41/results/retrieval-fidelity-001-qkv-preregister.{json,md}`. Execute rank0/rank1 CPU-only; persist before any next discriminator.
