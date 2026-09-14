# DS41-Q2-001 attempt034 — state/recovery and block-cost terminal

**Status:** `STATE_FAIL_B4_CORRUPT_FIRST / B2_CLEAN_COST_QUALIFIED / STOP`

> **Attribution correction after attempt036:** the attempt034 corrupt-first FAIL is preserved, but the earlier interpretation as latent dirty rollback/cache is withdrawn. Raw034 shows its final `[95,96,97]` forward had T=3: routed native remained active while mHC projection/RMS and coefficient/Sinkhorn fell back. Attempt036 changes only that missing T3 mHC dispatch math and the unchanged corrupt-first control becomes bit-exact on all 56 positions. Demonstrated cause: **T3 mHC dispatch gap**, not dirty cache. Raw attempt034 is unchanged.

## Recovery

Clean B1/B2/B4 fidelity remains exact on all 56 common-prefix positions per width (`rel-L2=0`, `max-abs=0`) on both ranks.

- `diagnostic-B4-corrupt-last`: **PASS**. One changed last draft is rejected (`num_sampled=3`, `num_rejected=1`) and all 56 compared positions recover exactly.
- `diagnostic-B4-corrupt-first`: **FAIL**. The first changed draft is rejected correctly (`num_sampled=1`, `num_rejected=3`) and the returned 64-token stream is still the oracle; the first observable numerical divergence appears at position 95: rel-L2 `0.0111638447`, max-abs `0.25`; positions 96 and 97 also fail. This is identical on both ranks.

Therefore B4 is not state-qualified. Its performance cost is **N/A**.

## Qualified clean cost

Times use the rank-coupled maximum wall for complete selected EngineCore steps (replay proposal bridge + scheduler/post_step + GPU completion), not isolated kernels. Prefill/TTFT and diagnostic tensor-copy requests are excluded from the steady cost gate.

| Path | Qualified samples | Work per block | Coupled mean | Median | Ideal verifier-only rate |
|---|---:|---:|---:|---:|---:|
| M1 | 72 steps | 1 new token | **79.0562 ms/token** | 78.1746 ms | 12.6492 tok/s |
| B2 clean | 36 blocks | 2 new tokens | **93.7432 ms/block** | 92.9332 ms | 21.3349 tok/s |
| B4 | N/A | 4-position packet | **N/A** | N/A | N/A |

Contemporary two-step M1 cost is `2 * 79.0562 = 158.1124 ms`. B2 therefore leaves **64.3692 ms per accepted 2-token block** before a real drafter and any additional integration overhead. Frozen trial margins were `66.2164`, `62.7169`, `64.1742 ms`; all are positive. This is a favorable verifier bound only, equivalent to an ideal verifier speedup of `1.6869x` under perfect acceptance and zero-cost drafting.

The clean B2 full step already includes the diagnostic replay proposal bridge; its steady proposal work averaged about `2.85 ms/block` on rank0 and `3.07 ms/block` on rank1. This is **not** real DSpark draft cost.

Per-row work retained by the fidelity control: routed native M1 expert path and TileLang mHC projection/RMS. Coefficient/Sinkhorn remains one fused `[T,24]` call; shared expert and TP all-reduce remain batched once.

## Decision

**B2 clean verifier has measurable headroom: 64.37 ms/two-token block.** This establishes that 12–13 TPS is not a demonstrated physical ceiling and that the corrected verifier itself does not consume the entire two-token M1 budget.

**At the time of attempt034 the speculative path was not qualified**, because corrupt-first B4 failed its frozen numerical state gate. Attempt036 later demonstrates that this was caused by the missing T3 mHC dispatch path and state-qualifies the same recovery protocol; attempt034 itself still contains no qualified B4 cost measurement. Real drafter cost, real acceptance distribution, qualified B4 cost and production integration cost remain unknown.

Cleanup: supervisor rc=0; NODE01/NODE02 `OFF_VERIFIED`; owner `NONE/OFF` at `2026-09-14T08:33:29Z`.
