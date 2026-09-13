# DS41 WO_B LLMM1: final A/B result

Date: 2026-09-13. Attempts022 (components and setup) and023 (complete model A/B).

**CORRECTNESS_PASS / SPEED_GATE_FAIL / NO_PROMOTION.**

Contemporary A averaged **11.70417 tok/s**; B averaged **11.89974 tok/s**. Gain: **1.671%**. B won all three pairs but did not meet the frozen 5% promotion threshold. Preserve attempt020's 12.08665 tok/s reference and leave `DS41_ATTN_WOB_LLMM1` unset/0.

## Configuration

Runtime commit: `c2a9acecac2959ebfe28f0ee180251621218d545`. Candidate commit: `b2ca80f213ff8d6cf830892e2f79dfd7167b1f1f`. Numerical baseline: `facf86486f7e78d8ea75b0fda21231c1981eaaae`. Epoch: `1789321415202393918`.

Same verified DenseFix weights and Engram2, TP2 attention/dense, EP2 routed MoE, RCCL Socket over USB4. Native HIP MoE, fused mHC coefficient/Sinkhorn and TileLang projection/RMS remain enabled in both arms. Only B uses LLMM1 for local BF16 WO_B M=1; each call keeps the original single TP all-reduce. No new quantization, speculative decoding, driver or dependency installation.

One load, excluded 32-token warmup per arm, then A1/B1/B2/A2/A3/B3. Each speed request uses the exact historical 35-token binary-search prompt, 128 actual output tokens, temperature 0, seed 1, context 4096, prefix caching off. Speed requests deliberately ignore EOS and finish at length, as in attempt020. Quality tasks allow natural stopping. Historical speed/coding prompts have reasoning none; the 100-doors prompt alone uses high.

## Six decision samples, actual execution order

| Run | Decode tok/s | TTFT s | Offline wall s | Major faults rank0 / rank1 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 9.82958 | 4.891 | 17.811 | 5493 / 6121 |
| B1 | 10.17837 | 4.905 | 17.382 | 3874 / 4894 |
| B2 | 12.73033 | 4.912 | 14.888 | 2 / 1 |
| A2 | 12.65870 | 4.883 | 14.916 | 0 / 0 |
| A3 | 12.62422 | 4.916 | 14.976 | 0 / 1 |
| B3 | 12.79051 | 4.880 | 14.809 | 0 / 0 |

| Statistic | A | B |
| --- | ---: | ---: |
| Mean decode tok/s | 11.70417 | 11.89974 |
| Median decode tok/s | 12.62422 | 12.73033 |
| Sample SD tok/s | 1.62353 | 1.49105 |
| CV | 13.87% | 12.53% |
| Mean TTFT s | 4.89659 | 4.89881 |
| Mean offline wall s | 15.90102 | 15.69320 |

Paired gains: +3.548%, +0.566%, +1.317%. Mean offline wall-throughput gain: 1.324%. HTTP metrics and isolated prefill TPS: N/A. Load/init 336.586s excluded.

Early requests have thousands of major faults; later speed requests have 0-2. These are whole-request counters, including prefill. Their source was not localized to weights, Engram, swap or decode. Combined with differing A/B text, they limit attribution to LLMM1 alone. No slow sample was discarded or replaced. The warm samples are descriptive, not a new qualification.

## Correctness and task outcomes

- Component gates PASS on both hosts: fixed rel-L2<=0.005/max-abs<=0.125, observed shard-sum rel-L2 0.00295466/max-abs 0.015625 on four canonical layer0 vectors. Reference uses the configured unquantized method, with F.linear equivalence recorded. These are reconstructed DenseFix intermediates; attempt021 saved zero attention samples.
- Dispatch gate: exactly one kernel and one reduction; 25 fallback cases and default-off verified. The component shard sum is arithmetic on one device, not a network test.
- Full-model execution: 24,560 LLMM1 calls on each rank, 320 fallbacks, exclusively M!=1. Every baseline request has zero candidate calls. All 12 requests have identical token streams between ranks. The native HIP library SHA is unchanged.
- A and B are not token-identical. Every speed pair first differs at zero-based completion index 28 (the 29th token). Rank agreement and this small task suite do not establish general intelligence or long-context equivalence.
- Arithmetic: 323, 2 tokens, natural stop. Coding: first_missing_positive, 101 tokens, natural stop, nine independent cases PASS for one in-place O(n)/O(1) function. JSON: correct object, 15 tokens, natural stop. Reasoning-high: 100 doors, answer 10, 88 tokens, natural stop. No output was repaired before validation.
- Runtime reports 80.37 GiB model allocation per rank. Process RSS high-water marks are 39.302/39.305 GiB and are not additive with GPU allocation on UMA. Per-request faults and minor-fault counts are retained in JSON.

## Setup negative and cleanup

Attempt022 produced no generation. vLLM model-class inspection aborted on conflicting core/devel ROCPROFILER_REGISTER_LIBRARY paths. The original import reproducer failed; core-first lookup passed on both hosts. NODE01's 29 overlapping core/devel shared libraries are hardlinks to the same inodes. The isolated launcher change fixes lookup order without changing dependency files. All attempt022 evidence remains preserved.

Attempt023 completed both ranks, then the supervisor verified both units/cgroups OFF and owner NONE/OFF at 17:52:27Z. GLM engine/coordinator remained OFF; its gateway stayed active. The inherited runner logs SystemExit(0) as a fatal event after run_complete; this is misleading exit logging, not an inference failure. The validator's overall FAIL means the performance threshold failed, while its correctness status is PASS.

## Evidence and exact continuation

Local component/setup root: `reports/DS41-Q2-001/attempt022/`.

- `preregister.json`, `prompt-tokens.json`, `component-gate.sh`.
- `component-node01.json`, `dispatch-node01.json`, `component-node01.log`.
- `node02/component-node02.json`, `node02/dispatch-node02.json`, `node02/component-node02.log`.
- `rank0.log`, `node02/rank1.log`, `offline-rank0.events.jsonl`, `node02/offline-rank1.events.jsonl`, `supervisor.log`.
- `reproduce-child-import.sh`, `child-import-original.log`, `child-import-core-first.log`, `node02/child-import-core-first-node02.log`.
- `upstream-check.md`: bounded sources audit; no demonstrated dual-gfx1151 replacement justified a download.

Local model-run root: `reports/DS41-Q2-001/attempt023/`.

- `preregister.json`.
- `prompt-tokens.json`.
- `start-receipt.json`.
- `offline-rank0.json`.
- `offline-rank0.events.jsonl`.
- `node02/offline-rank1.json`.
- `node02/offline-rank1.events.jsonl`.
- `rank0.log`.
- `node02/rank1.log`.
- `wob-ab-validation.json`.
- `validator.log`.
- `supervisor.log`.

Tracked compact result: [attempt023-wob-ab.json](attempt023-wob-ab.json). The live PLAN is external to this repository and was updated separately.

**NEXT:** no automatic rerun or second optimization. Keep the qualified baseline and the WO_B flag off. A future campaign should first distinguish cold/new-prompt page-in from warm decode before spending on another small GEMM substitution. This is a proposal, not a localized root cause or a guaranteed speedup. DSpark needs its own concrete gfx1151 and M>1 compatibility evidence; GLM DFlash qualification does not transfer.

To revalidate existing data without loading a model:

```bash
timeout 30 python3 scripts/validate-ds41-wob-ab.py \
  --rank0 reports/DS41-Q2-001/attempt023/offline-rank0.json \
  --rank1 reports/DS41-Q2-001/attempt023/node02/offline-rank1.json \
  --output /tmp/ds41-attempt023-revalidation.json
```

Expected exit 1: correctness PASS, speed FAIL, promotion false. Do not rerun attempt020, the completed attention profile, rejected primitive variants, attempt022's broken initialization or the same six decision samples unchanged.
