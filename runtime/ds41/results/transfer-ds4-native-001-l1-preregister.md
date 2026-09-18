# DS41 TRANSFER DS4 → NATIVE 001 — L1 preregister

Frozen before native model load. Candidate keeps existing DenseFix/Engram2 + MMQ + four-worker Engram reader + canonical-prefill + K2/M4; only chat input profile changes to `ds4-low-v1`.

One load, maximum six document requests, cap2048, temp0/seed1, no semantic retry, no prefix cache. If 6/6 PASS, proceed directly to L3 and do not import Antirez by principle. If L1 fails, preserve the negative and proceed to L2.

Promotion criterion is frozen now: candidate must be quality-eligible and show at least 10% median end-to-end improvement across code/docs versus a contemporary DS4 control, with no per-task regression beyond measured control noise. 200 prefill tok/s remains a target, not a gate promised in advance.
