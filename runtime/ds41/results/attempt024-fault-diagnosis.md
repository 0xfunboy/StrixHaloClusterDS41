# Attempt024: Engram miss-path I/O localized

Status: DIAGNOSIS_COMPLETE / ONE_CAUSAL_CANDIDATE_JUSTIFIED.
WO_B LLMM1 remains CLOSED, correctness PASS, speed gate FAIL, no promotion.
Numerical baseline `facf86486f7e78d8ea75b0fda21231c1981eaaae` is unchanged.
Diagnostic source: `1ebb0e2850ec81a190b39d9778323c4d31ffe56d`.

## Observed workload

One load, excluded P32 warmup, then P/P/Q/Q/P128 and a final clean P128
instrumentation control. P has35 input tokens, Q37. Temperature0, seed1,
4K context, prefix caching off, performance ignore_eos true. Prompt spec,
source and launch are frozen in the raw directory. No filesystem cache
reset was performed; this is not an OS-cold measurement.

| Rank0 request | TTFT s | First-to-last TPS | Offline wall s | Major faults r0/r1 | Physical process reads r0/r1 MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 | 4.8837 | 9.87655 | 17.7442 | 5493/6135 | 699.2/782.4 |
| P2 | 4.8820 | 12.60552 | 14.9583 | 1/0 | 0.004/0 |
| Q1 new | 5.4524 | 9.24149 | 19.1961 | 8430/8914 | 1057.1/1116.3 |
| Q2 | 5.0655 | 12.64031 | 15.1140 | 0/0 | 0/0 |
| P3 | 4.8179 | 12.48952 | 14.9878 | 0/1 | 0/0.004 |
| P clean control | 4.8243 | 12.67533 | 14.8439 | Not instrumented | Not instrumented |

Every measured request produced128 tokens. P repeats, Q repeats and all
seven rank pairs are token-identical, including the32-token warmup.
No deterministic-output change was introduced by the observer.
Warmup was9.112/9.133s, init331.701/330.743s, both reported separately.
HTTP and isolated prefill throughput are unavailable in offline SPMD.

## Source and phase

The source-tagged `SafeTensorMMap.affine2_rows` calls for
`layers.1.engram.embed` and `layers.14.engram.embed` account for roughly99%
of the request's process major faults on both nodes. The files are:

```
/home/funboy/models/ds41/engram2-tp2/rank0/model-00047-of-00048.safetensors
/home/funboy/models/ds41/engram2-tp2/rank0/model-00048-of-00048.safetensors
/home/funboy/models/ds41/engram2-tp2/rank1/model-00047-of-00048.safetensors
/home/funboy/models/ds41/engram2-tp2/rank1/model-00048-of-00048.safetensors
```

For P1, the reader is invoked only after the first token: total reader wall
1.9305/1.9150s, calling-thread CPU0.1568/0.1660s, reader major faults5426/6068.
Process first-to-last major faults5492/6134; before-first faults1/1.
Logical packed row bytes156800/174720, versus measured process disk reads
699.2/782.4MB. These are independently measured counters, not a conversion
from fault counts. Adjacent-page read-ahead is a plausible avoidable component
of this large amplification; individual fault addresses were not traced.

For Q1 the reader recurs in both phases: before-first0.3535/0.4157s,
decode2.7139/2.3603s, total3.0675/2.7759s. P1 minus P2 wall is2.7859s;
Q1 minus Q2 wall is4.0821s, of which0.3870s is TTFT difference and about3.695s
is first-to-last difference. Reader work explains a substantial localized
path, not an exact additive decomposition of the pair's critical path.
Reader elapsed includes gather/dequantization, scheduling and I/O waits;
it is not pure block-device wait. Times from the two ranks must not be added.

Repeats cause zero reader calls: the existing decoded-row LRU works.
Conclusion B: cost recurs with new requested data, not only once after load.
This does not make demand paging itself a bug. The single justified candidate
is scoped random-access advice to reduce unnecessary read-ahead on these tables.

## Other memory and observer evidence

Actual model processes are synchronous `InprocClient` external-launcher ranks,
not API clients. SELF includes every thread; live child PIDs are recorded but
their work is not attributed. About0.8GB process swap exists after load and
small swap-ins continue, so the report does not claim absence of swap.
Process swap decreases only hundreds of KiB on new requests, no global
swap-out is observed during measured requests, and host available memory
stays around32.5/33.5GiB. Primary request fault activity is localized to the
mapped Engram reader, not inferred from global counters.

Four maps/smaps snapshots retain235 library paths on each rank, with no new
file mappings after init. Engram mapped RSS grows with the first accesses.
JIT monitor was active; no inference-time compilation warning appears.
This is not proof that every possible late GPU preparation cost is zero.

Direct instrumentation overhead is0.033-0.061% of request wall, including
boundary collection. Maps snapshots cost78-99ms each outside requests.
P3 versus clean control differs by1.49% in decode, larger than direct observer
cost and not independently attributable from one pair. No timing is corrected
by subtracting observer overhead, and this diagnostic promotes no TPS record.

## Next and evidence

Test exactly one `MADV_RANDOM` candidate on full interior pages of the three
embedding tensors. Leave adjacent q/k/WKV/header pages, GGUF release, math,
precision, routing, KV and the65536-row LRUs unchanged. The
[Linux madvise documentation](https://man7.org/linux/man-pages/man2/madvise.2.html)
defines random advice as a performance hint without changing these file bytes.
Graceful fallback to normal advice is required. No extra row cache or preload.

Validation must start A and B with equally empty decoded LRUs and verified
nonresident full table pages. Only clean, identified Engram table ranges may
be discarded in this test harness, never global caches. Record preparation
cost, boundary-page exclusions, exact output equality and warm decode.

Pre-run attempt025 setup amendment: a synthetic ext4 reproducer retained
boundary folios after range-only discard. Validation therefore discards clean
cache for the two exact Engram source files, not global caches, while preserving
materialized sidecar linears. Candidate advice still excludes boundary pages.
The zero-residency gate is unchanged. The original failure is a regression test
in `scripts/test-ds41-engram-experiment.py` and is documented under attempt025.

Raw root: `reports/DS41-Q2-001/attempt024/`.
Files: `preregister.json`, `prompt-tokens.json`, `source-hashes.txt`,
`start-once.sh`, `start-receipt.txt`, `pre-start-status.txt`,
`offline-rank0.json`, `offline-rank0.events.jsonl`, `rank0.log`,
`fault_maps_*.json`, matching rank1 files under `node02/`,
`fault-summary.json`, `supervisor.log`, `cleanup-status.txt`.
Recompute the summary with `scripts/summarize-ds41-faults.py`.

Pair cleanup succeeded21:24:19Z, both nodes OFF_VERIFIED and owner NONE/OFF.
GLM engine/coordinator remain OFF, gateway remains active.
