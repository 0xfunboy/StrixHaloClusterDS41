# RETRIEVAL FIDELITY 001 — target-only M1 result

Same `5bdfed6` release/source, target-only M1 with no speculative config, exact frozen1571-token discriminator. One frontend setup-negative HTTP503 was proven not to reach the model; the corrected direct paired request ran once.

M1 result: prefill 92.416969s / 16.999043 tok/s, TTFT92.535083s, wall98.360220s, cache0; output `begin=17,middle=23,end=41` with file list `__init__.py, api.py, artifact.json, prompt.go, test-ds41-prefill-metrics.py, envelope_test.go`; semantic FAIL.

The K2 rollback result was also FAIL but `end=29` and a different file list. Therefore K2/speculative decoding is not a necessary cause of the retrieval failure. The failure signature is mode-sensitive; target model/runtime/weights/reference contract remains unresolved.
