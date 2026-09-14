# Attempt033 combined routed+mHC fidelity verification

Status: **FIDELITY PASS / STATE AND COST NOT RUN**.

Attempt033 used runtime source `d280c67d717e96ecf38964d12f2feb156d998b6b`, math fix `ce446ef62836980b4d4a3c9a62b4374ba3394c8d`: D1 unchanged promoted M1; B2/B4 enabled rowwise native routed plus rowwise mHC P1+C1. The live validator was historical and produced a dispatch-only false-negative because it still required every M>1 path to fall back. Validator correction `03f6699194bf4de1dc6cac960369614712f77d2e` changes **only** dispatch expectations according to the preregistered rowwise flags; oracle, output, coverage, finite/top1, rel-L2 `0.005`, max-abs `0.125` and state/reject gates are unchanged. Historical attempts027/028/031 remain FAIL under the corrected validator.

## Frozen fidelity result

Both ranks independently PASS D1, B2 and B4. Each width compares all **56/56** common-prefix positions. B2 and B4: **max rel-L2 0.0, max-abs 0.0, exact top1 and exact 64-token oracle output**. The 16 narrow layer0/layer1 boundaries for B2 and B4 are also all bit-exact on each rank. Canonical hashes over all saved logits records and narrow boundary tensors are identical rank0/rank1.

This establishes target fidelity for clean B2/B4. It does **not** establish rejection/recovery state correctness or block cost: the stale live validator marked widths ineligible before corrupt-first/corrupt-last controls, so those controls were not run, and only the three historical-format B1 measurements were taken. B2/B4 diagnostic wall/decode observations include tensor-copy overhead and are not qualified performance.

Initial supervisor arm used a relative script path and exited status127 before supervision; the pair remained ACTIVE/owned and was not restarted. A replacement supervisor with the absolute path was verified ACTIVE before results were accepted and later performed pair-safe cleanup `rc=0` at 2026-09-14T08:19:27Z. Final pair: both `OFF_VERIFIED`, owner `NONE/OFF`.

**NEXT:** one unchanged-math attempt with corrected validator to run B4 corrupt-first/corrupt-last rejection/rollback controls and, only if state gate passes, the already-frozen B1/B2/B4 block-cost measurements.
