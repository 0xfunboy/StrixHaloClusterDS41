# FINALIZE DS4 SOAK — live handoff

Updated: 2026-09-17 after same-source Engram serial control.
Phase: one parallel4 confirmation required for cache-regime attribution.

- Serial control epoch `1789639111974937319`, SAME release/source `k2-mmq-engram-9c13117` / `9c13117f...`, only `DS41_ENGRAM_READ_WORKERS=1`.
- code-2k COMPLETE: prefill `29.238172s / 54.312561 tok/s`, TTFT `29.606769s`, wall `31.974004s`, cache0; semantic FAIL (`result=47`).
- Serial pre-request mincore nearly empty: NODE01 ~1.18MB/sidecar, NODE02 ~1.38–1.72MB (<0.012%). Post request ~168.7–196.4MB.
- Initial parallel4: `20.848176s / 76.169734 tok/s`, observed save `8.389996s`, but pre-request mincore was not captured.
- Fixed next act: OFF once, SAME release with workers4, capture pre-request mincore, one frozen code-2k confirmation+validator. No worker sweep or MMQ replay.

---

Updated: 2026-09-17 after initial MMQ+Engram full-model request.
Phase: Engram same-source attribution control next; CED remains blocked.

- Live release: `k2-mmq-engram-9c13117`, source `9c13117f56fd81d03c8c610a5fb0148ccdc603b9`, epoch `1789636073826523275`, READY/idle at last check.
- Effective profile: K2/M4, canonical=1, MMQ=1, CED=0, MADV_RANDOM=1, Engram workers=4/min_rows=256.
- Request `mmq-engram-parallel-code2k-e1789636073826523275` COMPLETE: 1588 computed/cache0, prefill `20.848176s / 76.169734 tok/s`, TTFT `21.245709s`, wall `23.618051s`, decode `13.820452 tok/s`, finish=stop.
- Semantic validator FAIL: expected result52/middle `_ds41_artifact.py`; actual result58/`median_file=artifact.json`; first/last correct.
- Per-request Engram counters: N/A on this resident load; reader counters are in EngineCore only and no request-scoped export exists. No new profiler/capture was added.
- Focused real-sidecar gate is authoritative for reader work: 12,288 rows/table/rank at T1024; ~149.8–150.0MB read/arm; serial reader ~3.93s vs parallel4 ~1.01s; output SHA and bytes exact on rank0/rank1 layer1/14. Scoped POSIX_FADV_DONTNEED was used only in the focused gate; the full-model request is NOT labelled SSD-cold.
- Historical same-source MMQ gate remains OFF `91.934516s / 17.273164 tok/s` vs ON `29.191465s / 54.399463 tok/s`; do not replay it for stats.
- CED remains `BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT`; do not reopen or disable K2.

Next exact act:
1. Preserve current result/raw; supported whole-pair OFF once.
2. Reload the SAME `k2-mmq-engram-9c13117` source with only `DS41_ENGRAM_READ_WORKERS=1` changed; canonical/MMQ/K2/M4/MADV_RANDOM unchanged.
3. One frozen code-2k request, original validator, then compare same-source serial vs parallel full-model. Record OS page-cache state if observable; do not call either arm SSD-cold without evidence.
4. Continue compact quality panel on best admissible profile, then residual-cost decision/release gates. Soak only after quality qualification.

---

## Prior handoff history

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

## CED terminal gate

- `runtime/ds41/results/ced-k2-contract-blocker.{json,md}`: `BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT`, no model load.
- Config + DS4 pin confirm SWA128, KV `[2,8,14,20]`, index `[2,8,14,20,24,28,32,36]`.
- DS4 `8db1d1d` decoder suffix is admitted only for sweeps >=8192 and uses `1+(39-layer)*127`, not fixed 128 rows/layer. Frozen code-2k chunks 1023+565 do not enter it.
- Current K2 DSpark consumes target aux hidden `[37,38,39]` for every scheduled target token and precomputes context KV from all those rows. Skipping those deep target rows breaks the qualified K2 state contract; recomputing them removes CED work elimination.
- Decision: CED OFF/BLOCKED for current K2 profile. Continue Engram on MMQ ON predecessor; label candidate MMQ+Engram.

## Engram model-free gate

- Candidate source `7b7d55a542b63dca43fe3ae348a52c15cf397da3`.
- `runtime/ds41/results/engram-parallel-reader-gate.{json,md}`: `PASS_ADMIT_FULLMODEL_ENGRAM`.
- Reader candidate: MADV_RANDOM retained; 4 bounded read workers only when missing rows >=256; source-row sorted partitions; first-use LRU insertion and output scatter remain original order/multiplicity; cache cap remains 65,536 rows.
- Real 1024-token / 12,288-row sidecar gate on rank0/rank1 and layers1/14: output SHA and physical read bytes exact between serial and candidate. Cold miss-reader speedup range ~3.843x..3.856x; warm remains milliseconds.
- Fixture contract PASS on both nodes for duplicates, disordered IDs, repeats, first-use LRU and eviction.
- Pair is OFF/NONE after the completed MMQ-OFF control. Next: package this source with the already-qualified DS4 external-arena library, load MMQ ON + Engram4 once, run frozen code-2k integration and validator.
