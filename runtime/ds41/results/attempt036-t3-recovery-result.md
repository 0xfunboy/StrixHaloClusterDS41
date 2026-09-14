# DS41-Q2-001 attempt036 — T3 recovery causal verification terminal

**Status:** `PASS / T3_CAUSAL_FIX_CONFIRMED / B2_B4_STATE_QUALIFIED / NO_PERF_RERUN`

## Result

The original attempt034 corrupt-first protocol is preserved: same prompt, oracle, cap64, sampling and corruption index. It still rejects the first changed B4 draft (`num_sampled=1`, `num_rejected=3`) and still ends with the natural T=3 packet `[95,96,97]`.

With the narrowly qualified T3 mHC dispatch enabled, that tail now runs the promoted paths with zero fallback: routed native `120` calls/tokens, coefficient/Sinkhorn `80` fused calls / `240` tokens, projection/RMS `240` TileLang row calls/tokens. All **56/56** compared positions are bit-exact to M1 (`rel-L2=0`, `max-abs=0`), including positions 95-97.

The other state controls also PASS exactly on both ranks:

- B2 `diagnostic-B2-corrupt-only`: one changed draft, `num_sampled=1`, `num_rejected=1`, 56/56 exact.
- B4 `diagnostic-B4-corrupt-last`: `num_sampled=3`, `num_rejected=1`, 56/56 exact.
- Clean B1/B2/B4: 56/56 exact per width.

Canonical digests of every saved diagnostic logits stream (D1, clean B2/B4, B2 reject, B4 corrupt-first/last) are identical between rank0 and rank1. All returned token streams also match across ranks. `state_eligible_widths={2:true,4:true}` on both ranks.

## Causal attribution correction

Attempt034's FAIL was real, but its interpretation as persistent dirty rollback/cache is withdrawn. Raw attempt034 proves its only failed positions 95-97 were exactly one T=3 forward where routed native stayed active while mHC projection/RMS and coefficient/Sinkhorn both fell back with `tokens_not_1`. Clean T4 and corrupt-last T1 used promoted mHC and passed.

Attempt036 changes only the missing T3 coverage in the existing opt-in mHC fidelity wrapper; the unchanged corrupt-first request then becomes bit-exact end-to-end. **Demonstrated cause: missing T3 mHC dispatch coverage.** No cache patch was required.

## Scope

This run was deliberately `state_only=true`; no B1/B2/B4 cost measurements were repeated. Attempt034's qualified clean B2 margin (`64.3692 ms/two-token block`) remains separate historical evidence. No real drafter, DSpark integration, acceptance study, new kernel, weight, engine or production-default change was introduced.

Cleanup: supervisor `rc=0`, NODE01/NODE02 `OFF_VERIFIED`, owner `NONE/OFF` at `2026-09-14T09:23:04Z`.
