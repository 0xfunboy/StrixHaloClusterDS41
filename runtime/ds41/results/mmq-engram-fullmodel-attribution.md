# MMQ+Engram full-model attribution

- Same source/release: `9c13117f56fd81d03c8c610a5fb0148ccdc603b9` / `k2-mmq-engram-9c13117`; K2/M4 canonical=1 MMQ=1 CED=0 MADV_RANDOM=1.
- Serial reader1: 29.238172s / 54.312561 tok/s; TTFT 29.606769s; wall 31.974004s.
- Parallel reader4 confirmation: 23.792395s / 66.744018 tok/s; TTFT 24.171833s; wall 26.544321s.
- Attributable confirmation: 1.2289x by prefill time, 18.63% time reduction; TTFT saved 5.435s.
- Parallel samples: 20.848176s, 23.792395s; median 22.320285s. Serial / parallel-median time ratio 1.3099x.
- Prefix cache is zero on every full-model arm. Serial and confirmation parallel both had <0.012% whole-sidecar mincore before request; neither is labelled physical-SSD-cold.
- Focused sidecar gate: 12,288 rows/table/rank, ~149.8-150.0 MB read per arm, identical bytes/SHA; reader wall ~3.93s serial vs ~1.01s parallel4 (3.843-3.856x).
- Per-request LRU/reader counters remain N/A because the resident EngineCore has no request-scoped export; no values inferred.
- Semantic validator: serial FAIL, initial parallel FAIL, confirmation parallel FAIL. Engram performance gain is confirmed; quality remains open.
