# DS41 BLOCK_M4/8 full-model observed result

- Source: `bc41a355e827fbb193e65069cf867584cc9bb72f`.
- Component non-regression: **8/8 bit-exact M4/M8**.
- A/M4 prefill: **98.366508s**; B/M8: **82.476047s**.
- Observed reduction: **16.154341%** time; throughput ratio **+19.266759%**; **15.890461s** saved.
- A output: `{"result": 48, "first_file": "__init__.py", "last_file": "test-ds41-prefill-metrics.py", "median_file": "app.py"}`
- B output: `{"result": 46, "first_file": "__init__.py", "middle_file": "prompt.go", "last_file": "envelope_test.go"}`
- Both semantic **FAIL** against frozen code-2k expected.
- Full-model outputs differ; same-arm repeatability is not yet measured, so the difference is **not attributed to BLOCK_M8**.
- M8 remains experimental / not promoted.
- Historical `FAIL_NUMERIC_GATE_STOP` remains preserved; the shared `max_abs=128` was later localized to the fused-quantized-vs-explicit-reference numerical contract and is not an M8-only regression.
