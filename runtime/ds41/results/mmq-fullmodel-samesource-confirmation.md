# DS4 MMQ same-source full-model confirmation

Status: **MMQ speed isolated / quality unqualified**.

Execution order was B2 MMQ ON followed by one same-source MMQ OFF control; this is not an alternating A/B. Both use source `3b12582f868dd922fca2ffe5ebffc10aa922e4a4`, K2/M4, canonical-prefill ON, identical frozen code-2k input/sampling/cache policy.

- MMQ ON B2: **29.191465s / 54.399463 tok/s**, TTFT 29.596120s, wall 31.989323s.
- MMQ OFF control: **91.934516s / 17.273164 tok/s**, TTFT 92.357583s, wall 95.091196s.
- Same-source prefill speedup: **3.149363x**; time reduction **68.248%**; TTFT saved **62.761463s**; wall saved **63.101872s**.
- Semantic validator: B2 **FAIL**; control **FAIL**. MMQ speed is not a quality promotion and no decode gain is inferred.
- `DS41_DS4_MMQ_STATS` did not survive normal SIGTERM lifecycle because reporting is Python-`atexit` only. No aggregate/per-request counts are invented. External arena component gate remains PASS with 35,414,272B high-water and 0B leaked.

Decision: continue the already-authorized independent CED branch while carrying quality as an open gate.
