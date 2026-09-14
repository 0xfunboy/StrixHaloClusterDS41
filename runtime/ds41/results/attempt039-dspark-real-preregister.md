# DS41-Q2-001 — real DSpark K=1 A/B preregistration

Source starts after real-weight gfx1151 PASS. Two full loads are required because speculative decoding is an engine configuration; no hot-switch is introduced.

- B first: `attempt039-dspark-k1`, real `method=dspark`, K=1, draft TP2/EP2, all three MTP stages, hidden sources 37/38/39, adaptive verification OFF, standard greedy draft/rejection. `DS41_DSPARK_MXFP4_BF16=1` uses the qualified load-time exact expert fallback; target M>1 rowwise MoE/mHC controls stay enabled.
- A second: `attempt040-m1-control`, promoted M1 target only, same target/config/cache/sampling/prompt panel.
- Each load: speed warmup32 excluded, same speed prompt functional64 excluded, then three retained speed128 requests, then arithmetic/coding/JSON/reasoning-high natural-stop quality requests.
- B functional64 captures real proposed token IDs and native detailed acceptance; performance requests capture native acceptance + proposer timing without token copies or in-request synchronization.
- No replay/oracle/future continuation is available to the drafter. The validator compares final emitted token IDs A vs B, validates quality independently, preserves every speed sample, and requires a natural B rejection plus subsequent continuation before state recovery is considered demonstrated.
- Performance: full client wall/decode TPS/TTFT. Proposal CUDA-event span is reported as a component observation only and is never subtracted from wall or added to overlapping target work.
