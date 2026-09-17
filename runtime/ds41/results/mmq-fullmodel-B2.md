# DS4 MMQ full-model B2 checkpoint

Status: **PERFORMANCE VALID / SEMANTIC FAIL / STATS PENDING**.

- epoch `1789632423869537084`, source `3b12582f868dd922fca2ffe5ebffc10aa922e4a4`, release `k2-mmq-arena-3b12582`
- request `mmq-ab-B2-code2k-e1789632423869537084`; K2/M4, canonical-prefill ON, DS4 MMQ ON, external arena 64 MiB
- 1588 prompt tokens, cache0, HTTP200, natural stop
- prefill **29.191465s / 54.399463 tok/s**
- client TTFT **29.596120s**, wall **31.989323s**, decode **13.700491 tok/s**
- semantic validator **FAIL**: expected `52/__init__.py/_ds41_artifact.py/envelope_test.go`; actual model JSON was `{"result": 62, "first_file": "__init__.py", "last_file": "envelope_test.go", "median_file": "build.go"}`
- A2→B2 observed prefill throughput ratio **3.3171x** is cross-source and is not yet isolated MMQ effect.
- external-arena component recovery gate remains PASS; runtime MMQ coverage/stats are pending collection at the required lifecycle switch.

Next: whole-pair OFF only after confirmed idle, preserve both rank logs/stats, then run one same-source DS4-MMQ-OFF control. Do not resend B2.
