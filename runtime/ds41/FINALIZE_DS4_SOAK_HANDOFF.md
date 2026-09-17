# FINALIZE DS4 SOAK — live handoff

Updated: 2026-09-17 after same-source MMQ confirmation.
Phase: MMQ confirmation closed; CED next.

- Recovery source/release: `3b12582f868dd922fca2ffe5ebffc10aa922e4a4` / `k2-mmq-arena-3b12582`.
- B2 epoch `1789632423869537084`, request `mmq-ab-B2-code2k-e1789632423869537084`, MMQ ON: 1588/cache0, prefill 29.191465s / 54.399463 tok/s, TTFT 29.596120s, wall 31.989323s, semantic FAIL.
- Same-source control epoch `1789634721464248154`, request `mmq-control-off-3b12582-code2k-e1789634721464248154`, MMQ OFF: prefill 91.934516s / 17.273164 tok/s, TTFT 92.357583s, wall 95.091196s, semantic FAIL.
- Same-source MMQ prefill speedup **3.149363x**, time reduction **68.248%**. No decode gain claimed.
- B1 remains `FAILED_RUNTIME_MEMORY_GUARD`, no valid performance sample. A2 remains preserved historical cross-source control.
- B2 DS4 aggregate stats unavailable after supported OFF: runtime reports only at Python atexit while systemd OFF uses SIGTERM. No inferred counts. External arena gate remains PASS (64MiB cap; 35,414,272B observed high-water on real T1023 fixture; 0B in-use after call).
- Registry: `reports/DS41-Q2-001/mmq-ab-fullmodel/registry.json`.
- Confirmation: `runtime/ds41/results/mmq-fullmodel-samesource-confirmation.{json,md}`.

Live at handoff update: same-source MMQ-OFF control remains READY/idle on epoch `1789634721464248154`, source `3b12582`; pair poison empty. Do not resend B2 or control.

Next exact act:
1. Read mandate §4 CED and only relevant §36 references.
2. Implement bounded CED/SWA replay preserving source KV [2,8,14,20], indexer/compressor state, carry mHC, positions and hidden37/38/39; no simplistic last-128 truncation.
3. Model-free/state gate before a new model load; then one qualified integration measurement.
4. Continue Engram §5 even if CED independently blocks, using the best admissible predecessor and honest labeling.
5. Quality/repeated confirmation/release; soak only after gates.
