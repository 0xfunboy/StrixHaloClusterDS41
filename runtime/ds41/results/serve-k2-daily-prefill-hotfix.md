# SERVE-K2 daily prefill telemetry hotfix

Real daily smoke on epoch `1789467122418922391` proved `reasoning_effort=none` and exact final `323`, plus authoritative computed/cached prompt counters, but `prefill_engine_ms` was absent on both short and multi-token requests. The collector also exposed a separate client-only bug: reading `resp.fp` surfaced HTTP chunk framing and waited for keep-alive; it is corrected to `HTTPResponse.readline()` and terminates on `[DONE]`.

Runtime cause: the CUDA end-event was queried after sampling/bookkeeping and skipped when `query()` raced completion. A request could finish before a later step resolved the event. Hotfix: when the final-prompt end-event exists, synchronize that event once per request after sampling/bookkeeping only if still incomplete, then emit elapsed start-to-logits-ready. This is not per-token and does not change model math/sampling.

## V2 integration
Serving uses `VLLM_USE_V2_MODEL_RUNNER=1`. The active runner now records the start event from V2 `InputBatch` when `num_computed_prefill_tokens_np==0`, records the end after target `compute_logits` and before grammar/sampling on the final prompt chunk, resolves once per request after sampler launch, and attaches the span to `ModelRunnerOutput`. This preserves all prior prompt computed/cached accounting and does not synchronize per token.
