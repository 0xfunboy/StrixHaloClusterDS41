# TRANSFER DS4 → NATIVE 001 — L1 result

**FAIL / profile transfer alone is not sufficient.** Both mandatory original document requests exhausted the full 2048 completion budget inside reasoning and emitted zero final content; holdouts and confirmations were therefore not run.

| Task | Result | Prefill | TTFT | Decode | Wall | K2 acceptance |
|---|---|---:|---:|---:|---:|---:|
| code2k-v2 | INCOMPLETE_NO_FINAL | 21.547s / 75.60 tok/s | 21.566s | 16.14 tok/s | 126.99s | 86.72% (1299/1498) |
| docs2k-v2 | INCOMPLETE_NO_FINAL | 17.901s / 72.96 tok/s | 17.917s | 16.81 tok/s | 121.86s | 86.60% (1299/1500) |

The input contract is not in doubt: L0 proved full DS4 token-ID equality through the actual renderer. The native speculative path is also active and highly accepted. Therefore this negative does **not** isolate DSpark, MMQ, Engram2 or a kernel as the cause; it proves that the old DenseFix/Engram2 target recipe does not inherit DS4 document quality merely by receiving the same LOW input.

Per preregistration, L1 FAIL proceeds to L2: calibrated Antirez Q2 + native Engram, target-only first. Qualified DS4 is restored while L2 software preparation runs.
