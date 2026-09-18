# DS4 DOCUMENT PROFILE 002 — document quality

Status: **6/6 PASS** under the single frozen LOW profile (`DS4_THINK_LOW`, numeric level absent, cap2048 total).

| Case | Result | Prefill | Decode | First final | Wall | Cache |
|---|---|---:|---:|---:|---:|---:|
| code2k v2 | PASS | 20.478s / 79.55 tok/s | 14.41 | 64.349s | 66.180s | 0 |
| docs2k v2 | PASS | 25.570s / 51.08 tok/s | 15.84 | 43.196s | 45.703s | 0 |
| holdout A | PASS 10/10 | 18.600s / 75.75 tok/s | 15.21 | 59.554s | 64.087s | 0 |
| holdout B | PASS 10/10 | 22.294s / 91.46 tok/s | 15.73 | 65.168s | 70.021s | 0 |
| code2k confirm | PASS independent | 18.067s / 90.17 tok/s | 15.95 | 57.461s | 59.289s | 0 |
| docs2k confirm | PASS independent | 14.870s / 87.83 tok/s | 16.16 | 32.150s | 34.584s | 0 |

The historical NONE document FAILs remain preserved; this is a distinct LOW profile result. C-off, Go-low and R4 retrieval evidence were reused, not rerun. Six-pass admits the preregistered continuation in the same DS4 load.
