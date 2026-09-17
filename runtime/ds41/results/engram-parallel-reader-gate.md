# Engram bounded parallel reader gate

**Status: PASS_ADMIT_FULLMODEL_ENGRAM.**

Candidate source `7b7d55a542b63dca43fe3ae348a52c15cf397da3` keeps the promoted `MADV_RANDOM` policy and the 65,536-row decoded LRU. It changes only large miss reads: four bounded workers for miss sets >=256 rows, source-row sorted partitions, then insertion and scatter in the original first-use/order/multiplicity.

| Rank | Layer | Serial cold ms | 4-worker cold ms | Speedup | Read bytes equal | Output SHA equal |
|---:|---:|---:|---:|---:|---|---|
| 0 | 1 | 3949.660 | 1025.670 | **3.851x** | yes | yes |
| 0 | 14 | 3940.943 | 1024.245 | **3.848x** | yes | yes |
| 1 | 1 | 3938.822 | 1021.499 | **3.856x** | yes | yes |
| 1 | 14 | 3943.986 | 1026.342 | **3.843x** | yes | yes |

Cold reader speedup range **3.843x–3.856x**, mean **3.849x**. Read bytes and output SHA are identical arm-by-arm. Warm lookup remains a few milliseconds; its small absolute regression is retained, not hidden.

Contract tests also pass duplicate/disordered IDs, repeat hits, first-use LRU order and bounded eviction on both nodes. No model load was used for this gate.

The scoped cold preparation uses a fresh process plus `POSIX_FADV_DONTNEED` for the exact Engram file with `MADV_RANDOM`; it is not labeled a physical SSD-cold reboot.

Decision: admit one full-model **MMQ+Engram** integration measurement. CED remains blocked and is not part of this candidate.
