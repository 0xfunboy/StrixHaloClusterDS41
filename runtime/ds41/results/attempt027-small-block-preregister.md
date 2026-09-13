# Attempt027: small-block target verification, frozen gate

Status: pre-load protocol. Parent runtime977ffbf and numerical baselinefacf864
are preserved. This is a teacher-forced/replay feasibility experiment, not
DSpark integration or speculative production throughput.

## Source audit and scope

The actual launcher sets `VLLM_USE_V2_MODEL_RUNNER=1`. At pinned vLLM
`0bfb653d3b5161660db9ada0d84c2cdd60961de7`, ordinary `ngram` and
`custom_class` are rejected by `config/vllm.py:2635` and the V2 speculator
factory has no corresponding implementations. The legacy custom proposer is
not a usable hook in this runner. Config-only preflight confirms target-only
PASS, both methods rejected at K1/K3. No model was loaded for that probe.

A bounded, process-local diagnostic adapter admits exactly
`runtime.ds41.block_verify_experiment.ReplaySpeculator`, Kmax3, synchronous V2.
It supplies previously recorded tokens, not learned predictions. Its only
compatibility hooks are exact-class config admission, factory selection,
scheduler-facing proposal-list truncation, and a guard for V4.1's unused
optional None MTP hidden getter. The existing target, metadata, scheduler,
rejection sampler and caches are retained. No vendor source, weights, driver,
network, API or UI changes. Config-only bridge preflight PASS.

In `v1/worker/gpu/model_runner.py`, scheduled draft lengths select K+1 logits
rows and combine sampled/draft input tokens. Speculative widths are same-sequence
decode after a committed prefix, not independent requests or ordinary prefill.
The greedy rejection sampler emits accepted drafts plus one recovered/bonus
token. GPU request cursors and scheduler CPU `num_computed_tokens` both subtract
rejected suffix length. V4.1 Engram lookbacks are regenerated from corrected
cursors and committed token storage. The compressor ring is sized at least8
raw rows for Kmax3; SWA retention is based on processed tokens. These are source
contracts to test, not an assertion that reject/rollback already passed.

ROCm metadata rebuilds ragged SWA indices per target step. Its disabled
`supports_draft_decode_metadata_update` concerns fused draft multi-step metadata
updates; this diagnostic rebuilds target metadata normally. No bypass of causal
masking or RoPE is introduced. Full accept alone will not qualify recovery.

## Numerical dispatch being tested

| Component | M1 promoted | M2/M4 actual fallback |
| --- | --- | --- |
| Routed MoE | Native HIP, IQ2_XXS/Q2_K weights, Q8_1 activations | EP-aware Triton, original BF16 activations |
| mHC projection/RMS | M1 TileLang FP32 | Eager FP32 projection/RMS |
| mHC coeff/Sinkhorn | M1 fused Triton | Eager sigmoid/softmax/normalization loops |

Native MoE quantizes activations for gate/up AND down. Its Q8_1 block32
uses amax/127, rounded signed bytes and FP16 scale/sum. The Triton path does
not perform the same activation quantization. Packet-vs-serial equivalence
therefore cannot be assumed from shared weights or rank agreement. Simply
relaxing the M1 guard is unsafe: native wrapper token counts and reshapes are
specialized to one sequence token, with six routed rows for the down step.
No kernel extension is part of this gate.

## Frozen sequence and correctness

One supervised two-rank load, same existing MixedQ2 DenseFix/Engram2 artifacts,
TP2+EP, Socket USB4, eager, context4096, batch1, prefix cache OFF. Keep all
promoted flags1, including Engram advice, and WO_B0. No cache eviction campaign.
The35-token historical P prompt and first64 output tokens from attempt026
`engram-A1-Pfirst` are frozen in `attempt027/prompt-tokens.json`.

All requests are greedy temperature0/seed1. K remains0 until8 output tokens
have been emitted. A proposed verification block has B=K+1 positions: the last
committed anchor token plus K drafts. With perfect acceptance it yields at
most B NEW output tokens, not B+1. Oracle proposal acceptance is diagnostic,
not a forecast of real DSpark acceptance.

Sequence:

1. Excluded D1 warmup64, then D1 diagnostic64. Its outputs must exactly equal
   the preserved oracle on both ranks or the model-bearing gate stops.
2. Excluded B2 warmup64, B2 diagnostic64; excluded B4 warmup64, B4 diagnostic64.
3. If B4 clean fidelity passes, two diagnostic64 controls corrupt respectively
   the first or last draft in exactly one packet. Earlier-position logits must
   remain within the same gates; real rejection must occur and subsequent
   output/common-prefix logits must recover. No forced acceptance.
4. Three clean32-token measured trials for contemporary M1 and each eligible
   block width. Order1/2/4,4/2/1,1/2/4, omitting failed widths. B4 recovery
   failure excludes its timing qualification. All shape warmups and diagnostic
   tensor-copy runs are excluded from performance statistics.

Each compared position must have identical committed prefix/input tokens,
finite full real-vocabulary logits, exact top1, relative-L2<=0.005 and
max-absolute error<=0.125. Errors are reduced in FP64 from captured raw FP32
pre-sampler logits. These are conservative predeclared diagnostic tolerances,
not an established general whole-model error guarantee. No averaging away
failing positions or post-result relaxation. Compare rank agreement separately.
The changed draft's later positions are not common-prefix comparisons. Require
actual rejected counts, cursor rollback and exact recovered64-token output.

## Timing, budget and limits

Time complete in-process engine get_output: scheduler/metadata, target forward,
collectives, logits, sampling, post_step and GPU completion. CPU event logging,
counter snapshots and replay proposal work are disclosed. No per-operator
GPU synchronization. Diagnostic input/logit host copies are separate from
clean timing. The synchronous replay proposal cost remains included rather
than optimistically subtracted, and is recorded per step.

Use only steady decode steps after activation with actual width B, not prefill
or a shortened tail. Record total block wall, actual emitted tokens, kernel
dispatch counters, allocation/reservation and process/host memory snapshots.
No per-kernel dominance in milliseconds is inferred merely from call counts.

With contemporary serial cost t1 and measured complete replay-step cost tB,
the favorable zero-real-drafter/perfect-acceptance rate is B/tB and relative
factor B*t1/tB. Remaining budget for actual drafting, commit/rollback and
extra integration overhead is B*t1-tB. If nonpositive there is no measured
budget. Real drafter cost and acceptance remain N/A. The replay overhead
included in tB makes this an idealized bound using the measured diagnostic
path, not achieved DSpark TPS. Failed fidelity cannot establish readiness.

RuntimeMax1500s per rank and existing pair-safe supervisor1620s. Any rank
failure triggers paired cleanup. GLM remains OFF; gateway available. No new
model/download/framework or automatic full DSpark port. Final source pin is
recorded before launch. Raw and all negatives remain in
`reports/DS41-Q2-001/attempt027/`.
