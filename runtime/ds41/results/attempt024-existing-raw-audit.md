# Existing-data audit before any new load

Source: attempt023 rank0/rank1 JSON, events, and the frozen offline runner.
WO_B is closed: correctness PASS, speed gate FAIL, no promotion or rerun.

## Counter scope

The collector used `getrusage(RUSAGE_SELF)` in the SPMD process, including
its threads but excluding children and the peer. V1 multiprocessing was
disabled for the external launcher, so this was the process hosting the
local model, not an unrelated HTTP client. Deltas bracket `run_generation`,
including SamplingParams construction, enter/exit logging and result
extraction. The reported client wall instead brackets only `llm.generate`.
Arm switching and later checkpoint writes are outside the counter interval.
Peak RSS is a lifetime high-water mark. GPU allocation is a separate metric.

Both A1/A2/A3 and B1/B2/B3 have identical 128-token outputs within their arm
on each rank. All 12 requests match across ranks. All 24 generation-exit
events agree with the JSON timestamps/counts. A and B differ in text and
are not used as a perfectly controlled memory comparison.

| Request | Rank0 TTFT s | Decode span s | Wall s | Major faults rank0/rank1 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 4.891 | 12.920 | 17.811 | 5493/6121 |
| A2 | 4.883 | 10.033 | 14.916 | 0/0 |
| A3 | 4.916 | 10.060 | 14.976 | 0/1 |
| B1 | 4.905 | 12.477 | 17.382 | 3874/4894 |
| B2 | 4.912 | 9.976 | 14.888 | 2/1 |
| B3 | 4.880 | 9.929 | 14.809 | 0/0 |

A1 minus mean(A2,A3): rank0/rank1 decode excess 2.873858/2.873764s;
TTFT difference -0.008649/+0.002905s. B1 minus mean(B2,B3): decode excess
2.524735/2.525140s; TTFT difference +0.008997/+0.007922s. This localizes
elapsed-time excess after the first token, not the fault timing or cause.

New quality requests again fault: arithmetic 789/835, coding 6480/7585,
JSON 4435/4916, reasoning 7298/7799. These requests differ in lengths and
have no repeated controls, so the cost of prompt novelty cannot be quantified.
Excluded 32-token warmups also faulted: A 4576/4692 and B 215/225.

## What is known and what is missing

- Low faults on repeated speed requests do not imply later requests are warm.
- No count-to-byte or count-to-time conversion is justified.
- Prefill/decode fault split: **not determined** from these raw counters.
- Backing file, swap involvement, causal I/O stall time, child-process work
  and GPU fault coverage: **not determined**.
- Existing source candidates can be observed without changing behavior:
  `SafeTensorMMap.affine2_rows` gathers weight/scale/bias regions of the
  rank-local Engram sidecars; LRU hit/miss/rows_read counters already exist.
  Their presence does not establish causation.
- GGUF source discard is a proven loader fix and will remain unchanged.

Missing observation: phase snapshots on actual model outputs, process I/O
and memory state, plus source-tagged reader time/faults and cache deltas.
One baseline-only diagnostic load is justified. No new load preceded this audit.
