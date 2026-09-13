# Attempt025: setup abort, no benchmark samples

Source `083967e40159b664e59daf40b63f512430f697cb`,
epoch1789335540462395650. LLM initialization completed in329.015s on rank0;
then the experimental harness raised `expected exactly the two model Engram
embeddings` before any generate call. Both ranks were cleaned21:59:06Z,
owner NONE/OFF. No TPS or candidate correctness result exists for this attempt.

The harness incorrectly relied on `gc.get_objects()` to discover model objects.
Pinned vLLM freezes the GC heap after model construction in
`v1/engine/core.py:246` and `v1/worker/gpu_worker.py:927`. Frozen objects are
not returned by that enumeration. This is a harness setup defect, not a model
or candidate failure.

The correction follows the existing in-process ownership path:
`llm.llm_engine.model_executor.driver_worker.get_model().modules()`.
It retrieves both embedding modules and their two retained sidecar sources
explicitly. The production GC configuration is not changed or unfrozen.
Source-cache preparation receives those known sources instead of scanning GC.

Attempt026 is the setup retry. Candidate MADV_RANDOM ranges, numerical baseline,
prompt hashes, sampling, quality caps, order, metrics and gates are unchanged.
No additional performance candidate is introduced. The paired supervisor and
source-cache zero-residency gates remain required.

Raw: `reports/DS41-Q2-001/attempt025/`, including `rank0.log`,
`offline-rank0.events.jsonl`, matching rank1 files in `node02/`,
`start-receipt.txt`, `supervisor.log`, `cleanup-status.txt`, `preregister.json`,
`source-hashes.txt`, and `CPU-PREPARATION-NEGATIVE.md`.
