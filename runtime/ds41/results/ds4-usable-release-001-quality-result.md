# DS4 USABLE RELEASE 001 — quality result

Status: **DOCUMENT_QUALITY_FAIL / C_THINKING_OFF_PASS / NO_RELEASE_PROMOTION**.

| Gate | Result |
|---|---|
| code2k lower-middle v2 | **FAIL**: result52 and first_file correct, but middle=`prompt.go` and last=`test-ds41-prefill-metrics.py`; expected unchanged |
| docs2k lower-middle v2 | **FAIL**: result52 + first/last correct, middle=`README.md` instead of lower-center `pull_request_template.md` |
| C thinking-off | **PASS first attempt, zero repair**; thinking=false, effort none, 0 reasoning chars; generated complete replacement passes frozen sanitizer/private tests |
| Historical R4 retrieval/holdout | Preserved PASS evidence; not rerun |
| Historical C low | Preserved INCOMPLETE; not rewritten |

Observed diagnostics from mandatory quality calls, not a promoted benchmark:
- code2k v2: live prompt1629, cache0, prefill18.339s / 88.83 tok/s, first final18.599s, decode8.26 tok/s, wall23.322s.
- docs2k v2: live prompt1306, cache0, prefill28.086s / 46.50 tok/s, first final28.813s, decode15.18 tok/s, wall31.756s.
- C-off: live prompt597, cache0, prefill11.843s / 50.41 tok/s, first final11.961s, decode14.83 tok/s, wall31.573s, 292 completion tokens, natural stop.

The preload CLI token estimates were consistently +6 versus live server usage (1635→1629, 1312→1306, 603→597). This is preserved as an audit mismatch; the product tokenizer-sidecar preflight is not treated as live-server equivalence evidence.

Frozen decision: document quality gate failed, therefore P1/P2/P3 performance, context4K→16K, protected-gateway delivery and soak are **NOT_ADMITTED**. C thinking-off is qualified only for this specific fixture/profile. Restore K2.
