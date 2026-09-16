# DS41 canonical-topk full-model result

- Candidate: `85a51c77aef430d391c1ca709f2e1622b51b7967`, K2 / BLOCK_M4 / canonical sparse-prefill top-k enabled.
- **Initial raw target logits repeatability: PASS, bit-exact** request0/request1 on rank0 and rank1; cross-rank exact for each request.
- **Full output repeatability: FAIL**. API token IDs share 32-token prefix and first diverge at zero-based index 32; finish is `stop` in both.
- **Semantic code-2k: FAIL** on both through the original validator. Both return `result=46`, but other fields differ; expected result remains 52 with frozen expected files.
- Canonicalization executes 13 calls/request/rank: 5x shape 1023x512 and 8x shape 565x512, 9635 rows total. Component median extrapolation is ~1.8085 ms/request, **not** measured full-model overhead.
- Ordinary observed prefill: request0 96.0285s; request1 82.9509s. Their difference is not attributed to the sort.
- Interpretation: the canonical fix causally removes the demonstrated prefill first-logit variability, but residual decode/speculative-path variability remains after token 32. Do not extend the sort to decode/spec-verifier without a separate causal mandate.
- M8 remains deferred and unpromoted.

## Final restore

Known release `k2-prefill-5bdfed6` / source `5bdfed698...` restored on epoch `1789555810773881374`. Rank0/rank1/coordinator/gateway healthy, lifecycle `READY`, pair idle/poison empty. Authenticated gateway smoke returns exact `323`, reasoning0, natural stop. The resident release contains neither canonical-topk nor diagnostic capture hooks.
