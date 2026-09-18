# DS4 DOCUMENT PROFILE 002 — performance

Three independent PASS/cache0 code2k LOW samples, using two already-qualified document requests plus one additional comparable sample:

- 79.55 tok/s prefill, 14.41 decode
- 90.17 tok/s prefill, 15.95 decode
- 91.01 tok/s prefill, 16.64 decode

Median prefill: **90.17 tok/s**. Sample SD: **6.39 tok/s**. The 200 tok/s project target is **not met**.

These numbers describe the complete DS4 + calibrated Antirez Q2 + native DS4 Engram contract, not an engine-only delta.
