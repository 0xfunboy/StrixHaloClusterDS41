# DS41 M4 same-arm repeatability checkpoint

- Status: **M4_SAME_ARM_NONREPEATABILITY_OBSERVED**.
- Source `bc41a355e827fbb193e65069cf867584cc9bb72f`, epoch `1789544017915245827`, BLOCK_M4.
- Frozen request SHA `4926fd629b1b2057762c617c985e64211a9853c2678260758ee8ea4e09a364fd`.
- `M4-r1` is preserved as `CLIENT_CUTOFF_UNOBSERVED` and does not count.
- `M4-r1b`: `{"result": 47, "first_file": "__init__.py", "last_file": "test-ds41-prefill-metrics.py", "middle_file": "build_perfil_timing_metrics.py"}`
- `M4-r2`: `{"result": 48, "first_file": "__init__.py", "median_file": "prompt.go", "last_file": "envelope_test.go"}`
- Same-arm content equality: **FAIL**. First emitted token remains the same in both complete requests, while API-reported first-token logprob/top2 margin changes.
- M8×2 is deferred; prior M4/M8 text divergence cannot by itself be assigned to BLOCK_M8.
