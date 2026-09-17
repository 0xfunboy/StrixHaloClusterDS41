# FINALIZE DS4 SOAK — live handoff

Updated: 2026-09-17 after B2 terminal.
Phase: MMQ full-model confirmation; B2 complete, same-source OFF control pending.

- Repo `/home/funboy/StrixHaloClusterDS41`, branch `exp/ds41-q2-001`.
- B2 source/release: `3b12582f868dd922fca2ffe5ebffc10aa922e4a4` / `k2-mmq-arena-3b12582`.
- B2 epoch `1789632423869537084`; request `mmq-ab-B2-code2k-e1789632423869537084`.
- Profile: K2/M4, canonical-prefill ON, DS4 MMQ ON, external arena 64 MiB, RuntimeMax=infinity.
- B2 raw: `reports/DS41-Q2-001/mmq-ab-fullmodel/B2-code2k/`.
- B2 result: 1588/cache0, HTTP200, prefill 29191.46484375 ms = 54.399463 tok/s, TTFT 29.596120 s, wall 31.989323 s, decode 13.700491 tok/s, natural stop.
- B2 semantic validator: FAIL. Expected result52 + first `__init__.py` + middle `_ds41_artifact.py` + last `envelope_test.go`; actual JSON result62 + first `__init__.py` + `median_file=build.go` + last `envelope_test.go`.
- B2 is a valid performance sample, not quality-qualified.

Preserved history:
- A2 epoch `1789618441080477567`, source `ff582d652796e5e9fff2b1ecd488f48d0c06acbe`, MMQ OFF, COMPLETE: 96.829711s / 16.399925 tok/s, TTFT97.522224s, wall100.187667s, semantic FAIL. Do not resend.
- B1 epoch `1789624782020877084`, DS4 ON, `FAILED_RUNTIME_MEMORY_GUARD`; no valid performance sample. Do not replay B1 ID.
- B2 recovery gate `runtime/ds41/results/mmq-external-arena-gate.{json,md}` PASS on both nodes: T1023 x3 repeats exact, high-water35,414,272B, in-use0B.

Current lifecycle at checkpoint creation:
- epoch B2 remained READY, ranks/coordinator 200/200/200, pair idle, poison empty.
- `DS41_DS4_MMQ_STATS` not yet present in live logs; registry marks stats pending lifecycle OFF.
- Authoritative A/B registry: `reports/DS41-Q2-001/mmq-ab-fullmodel/registry.json`.
- Compact B2 result: `runtime/ds41/results/mmq-fullmodel-B2.{json,md}`.

Next exact act:
1. Confirm pair still idle; preserve/copy both rank logs before replacing transient units.
2. Use normal whole-pair OFF. Collect any terminal DS4 MMQ stats emitted by each rank; do not claim per-request attribution for aggregate counters.
3. Start same source/release `3b12582` with K2/M4, canonical=1, DS4 MMQ=0, same frozen code-2k request. Record actual order B2→OFF-control; this is not an alternating A/B.
4. Validate existing control raw, compare same-source timings, and integrate into the already-authorized confirmation. No redundant repetitions.
5. Continue CED, then Engram; soak remains final and gated.
