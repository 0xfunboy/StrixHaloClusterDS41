# DS41-Q2-001 attempt041/042 — real DSpark K2 vs K1 terminal result

**Status:** `PASS / EXPERIMENTAL_DSPARK_K2_QUALIFIED / PRODUCTION_DEFAULT_UNCHANGED`

Frozen validator: **PASS** (`QUALIFY_K2_EXPERIMENTAL`), SHA256 `9779561b82e527f3717378635fb319a8d7ae5a33d16a2e2cb92604cb2611538f`.

## Performance

K1 speed128: `15.70072720`, `17.42281737`, `17.38138823` tok/s; mean **16.83497760**, median 17.38138823, sample SD 0.98250805, CV 5.8361%.

K2 speed128: `17.94646200`, `20.25174629`, `20.09723931` tok/s; mean **19.43181587**, median 20.09723931, sample SD 1.28867187, CV 6.6318%.

Contemporary ratio-of-means gain: **+15.4253%**; K2 wins 3/3 same-index trials and the frozen >=5% gate. Mean offline wall 12.35778s -> 11.37036s (+7.9903% reduction; +8.6842% wall-throughput gain). Mean TTFT 4.79600s -> 4.81449s (+18.49ms). Distinct coding+text+reasoning total wall 38.37850s -> 36.72434s (-4.3101%).

## Correctness

All K1/K2 frozen greedy requests are token-, text-, finish-reason- and stop-reason-identical; every first-diff index is null. Rank0/rank1 coherence PASS for both arms. Arithmetic, exact JSON, reasoning-high and the distinct text prompt PASS. Independent saved-code validator PASSes **9/9** cases for both K1 and K2. Target dispatch switches remain active; target numerical math was not changed or recaptured, so attempt036 remains the frozen logits/state qualification.

## K2 acceptance

Across three speed128 repetitions: **225/312 = 72.1154%** draft-token acceptance over 156 verify calls. Full-width K2 block histogram accepted 0/1/2 = **24 / 39 / 93**. First proposal acceptance = **84.6154%**; second conditional on first = **70.4545%**; fully accepted blocks = **59.6154%**. No speed tail occurred; one-draft tail accounting was qualified model-free before load.

K1 contemporary speed acceptance: **174/210 = 82.8571%**, 210 verify calls. Final emitted tokens per verify call: K1 **1.82857**, K2 **2.46154**.

## Proposer cost

K1: **12.003ms/call** GPU-stream span, 213 calls, 2556.619ms total.

K2: **13.958ms/call** GPU-stream span, 159 calls, 2219.309ms total. Per-call cost is +16.29% but total proposer GPU span is -13.19% because K2 needs fewer proposal/verify cycles. CPU enqueue is overlapping/non-additive and is not summed into wall.

## Memory / lifecycle

Both K1 and K2 report **93.85 GiB/rank model-load footprint**. K2 reuses the same three trained DSpark stages, sidecar and BF16 expert fallback; no new weights or residency expansion is visible. B041 and A042 were separate supervised loads in order K2 -> K1. Both supervisors cleaned `rc=0`; final pair is `OFF_VERIFIED/OFF_VERIFIED`, owner `NONE/OFF`.

## Decision

**Qualify an experimental `dspark-k2-gfx1151` preset with explicit rollback to the already-qualified K1 preset.** Production default remains M1. No K3/K4, kernel, quantization or target changes are authorized by this result.
