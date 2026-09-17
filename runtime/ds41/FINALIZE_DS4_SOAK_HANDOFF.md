# FINALIZE DS4 SOAK — live handoff

Updated: 2026-09-17 after Engram same-source attribution confirmation.
Phase: compact quality panel; soak BLOCKED pending quality.

## Live candidate

- Source/release: `9c13117f56fd81d03c8c610a5fb0148ccdc603b9` / `k2-mmq-engram-9c13117`.
- Current epoch: `1789639579678586954`.
- Effective profile: K2/M4, canonical-prefill=1, MMQ=1, CED=0, MADV_RANDOM=1, Engram workers=4/min_rows=256, RuntimeMax=infinity.
- CED remains `BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT`; do not reopen or disable K2.

## MMQ established

- Same-source MMQ OFF control: 91.934516s / 17.273164 tok/s, TTFT92.357583s, wall95.091196s, semantic FAIL.
- MMQ ON B2: 29.191465s / 54.399463 tok/s, TTFT29.596120s, wall31.989323s, semantic FAIL.
- MMQ effect: 3.149363x throughput / -68.248% prefill time. Do not replay for lost atexit stats.
- External arena: 64MiB cap; component high-water35,414,272B; this is not total runtime memory.

## Engram attribution

- Serial workers1 control, epoch `1789639111974937319`: prefill29.238172s /54.312561 tok/s; TTFT29.606769s; wall31.974004s; cache0; semantic FAIL.
- Initial workers4: prefill20.848176s /76.169734 tok/s; TTFT21.245709s; wall23.618051s; cache0; semantic FAIL; pre-request OS residency unobserved.
- Workers4 confirmation, epoch `1789639579678586954`: prefill23.792395s /66.744018 tok/s; TTFT24.171833s; wall26.544321s; cache0; semantic FAIL.
- Serial vs characterized workers4: 1.228887x by prefill time, -18.626% time, +22.889% throughput, TTFT saved5.434936s.
- Two workers4 prefill samples median22.320285s; serial/parallel-median ratio1.309937x.
- Serial and confirmation pre-request whole-sidecar mincore are both <0.012%; neither is called physical SSD-cold.
- Focused reader gate: 12,288 rows/table/rank, ~149.8-150.0MB read/arm, same bytes/output SHA; serial read-wall ~3.93s vs parallel4 ~1.01s (3.843-3.856x).
- Per-request LRU/reader counters N/A: no request-scoped EngineCore export. Do not infer them.
- Evidence: `runtime/ds41/results/mmq-engram-fullmodel-attribution.{json,md}` and `reports/DS41-Q2-001/mmq-engram-fullmodel/registry.json`.

## Decision / next exact act

1. Performance: `ENGRAM_PERF_GAIN_CONFIRMED`; target200 not met (current best measured workers4 prefill 20.848s, characterized confirmation23.792s).
2. Quality: OPEN. All code-2k samples remain FAIL; no promotion/soak.
3. Execute compact mandated §6 quality panel only on current best admissible MMQ+Engram profile, with frozen fixtures/validators/expected values.
4. Use prescribed discriminant for remaining FAILs; do not reopen generic decode determinism.
5. Then update residual time balance and consider only the single authorized structural residual intervention if quality permits.
6. Update PLAN §2/§17/§36 and this handoff after every terminal result.
