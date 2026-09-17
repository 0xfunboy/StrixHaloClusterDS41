# MMQ + Engram initial full-model result

- Source/release: `9c13117f56fd81d03c8c610a5fb0148ccdc603b9` / `k2-mmq-engram-9c13117`.
- K2/M4, canonical ON, MMQ ON, CED OFF, MADV_RANDOM ON, Engram workers=4/min_rows=256.
- code-2k: 1588 computed, cache0; prefill 20.848176s / 76.169734 tok/s; TTFT 21.245709s; wall 23.618051s; decode 13.820452 tok/s.
- Semantic validator FAIL: expected result52 / middle `_ds41_artifact.py`; actual result58 / `median_file=artifact.json`; first/last correct.
- Per-request Engram LRU/read counters N/A: current resident process exposes no request-scoped export. Do not infer them. The focused reader gate retains exact bytes/timings and output-SHA evidence.
- Initial parallel request OS page-cache state was not snapshotted, therefore this is not labelled SSD-cold.
