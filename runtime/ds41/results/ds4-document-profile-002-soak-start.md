# DS4 DOCUMENT PROFILE 002 — soak started

Persistent soak is **IN_FLIGHT** under owner `DS4_DOCUMENT_PROFILE_002_20260918`.

- unit: `ds4-document-soak.service`
- invocation: `8d11fc92446740aebbc36cde819fae1c`
- requirement: at least 7200 real seconds and 24 sequential qualified document requests
- first real request: code2k PASS exact, HTTP200, wall72.961s, cache0, natural stop
- runtime at start: DS4 READY, K2 OFF, protected gateway and corrected tokenizer sidecar active

The soak runner persists every case. Any failure writes a terminal FAIL and invokes owner-bound K2 rollback. A complete PASS writes terminal PASS and finalizer `QUALIFIED`, leaving DS4 READY.
