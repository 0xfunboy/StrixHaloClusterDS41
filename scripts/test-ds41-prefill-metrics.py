#!/usr/bin/env python3
from vllm.v1.metrics.stats import RequestStateStats
from vllm.entrypoints.generate.base.serving import build_per_request_timing_metrics
s=RequestStateStats()
s.queued_ts=10.0; s.scheduled_ts=10.1; s.first_token_ts=10.7; s.last_token_ts=11.7
s.ds41_prefill_engine_ms=500.0
s.ds41_prompt_tokens_computed=2000
s.ds41_prompt_tokens_cached=48
s.ds41_prompt_tokens_local_cache=48
s.ds41_prompt_tokens_external_cache=0
s.ds41_prompt_tokens_cache_creation=32
m=build_per_request_timing_metrics(s, 21)
d=m.model_dump()
assert d['time_to_first_token_ms'] > d['prefill_engine_ms']
assert abs(d['prefill_engine_tokens_per_second']-4000.0)<1e-9
assert d['prompt_tokens_computed']==2000 and d['prompt_tokens_cached']==48
assert d['prompt_tokens_local_cache']==48 and d['prompt_tokens_cache_creation']==32
print('PASS',d)
