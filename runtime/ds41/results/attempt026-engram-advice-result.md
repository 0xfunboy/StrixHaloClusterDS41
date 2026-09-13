Origine dei fault: accessi alle due tabelle affine Engram, localizzati per file e fase nel reader; circa il 99% dei major fault osservati nella diagnosi024.
Costo attribuito: il percorso reader del nuovo Q costa 2,61/3,51s per rank con advice normale; la correzione recupera 1,73s di durata media della richiesta, senza equiparare reader wall a pura attesa disco.
Ricorre sui prompt nuovi: sì, anche a modello già caricato; le ripetizioni usano la cache delle righe decodificate.
Intervento: MADV_RANDOM sulle sole pagine intere interne a weight/scales/biases Engram, qualificato e promosso, con rollback all'advice normale.
Prima richiesta prima/dopo: P misurato dopo warmup esclusa, mediana 18,056 -> 16,675s; prompt nuovo Q, 19,594 -> 17,658s.
Decode caldo prima/dopo: P 12,6494 -> 12,6686 TPS; Q 12,6410 -> 12,6619 TPS, sostanzialmente invariato.

# Attempt026: qualified Engram miss-path I/O reduction

**Verdict: CORRECTNESS_PASS / SPEED_GATE_PASS / PROMOTION.**
The change reduces new-data request latency. It is not a sustained warm-decode
speedup or a new HTTP/API qualification. The historical12.08665TPS reference
and numerical baseline `facf86486f7e78d8ea75b0fda21231c1981eaaae` are preserved.
WO_B LLMM1 remains CLOSED: attempt023 correctness passed, speed gate failed,
no promotion and no further tuning.

## Cause and single intervention

[Diagnosis024](attempt024-fault-diagnosis.md) connects request faults to
`SafeTensorMMap.affine2_rows()` reading the two compact Engram SafeTensors
files, not merely to a cumulative process counter. On a new prompt the
source-tagged reader is active before the first token and during decode.
Repeats make no reader calls. This is recurring new-data cost, not exclusively
initialization or a one-time post-load penalty.

The only production intervention is scoped `mmap.MADV_RANDOM`. It discourages
read-ahead for sparse accesses to `layers.1.engram.embed` and
`layers.14.engram.embed`. Only complete pages inside packed weight/scales/biases
are advised; headers, boundary pages and q/k/WKV regions are excluded.
The [Linux advice contract](https://man7.org/linux/man-pages/man2/madvise.2.html)
does not change these tensor bytes. The reader validates tensor type, layout,
scope and ranges, records effective status, and attempts normal-advice fallback
on failure. Unsupported platforms retain the normal path.

No new decoded-row storage is added: each existing LRU retains its65536-row
limit. No weights, arithmetic, precision, routing, KV state, GGUF source discard,
driver, dependency, network or kernel configuration changes. Native HIP MoE,
mHC coefficient/Sinkhorn and projection/RMS remain1; WO_B remains0. Core-first
ROCm library lookup remains intact. No global cache drop, swapoff, model preload
or mlock was used.

## Frozen comparison and provenance

- Tested source: `f6b4ba737f7eab26103a97b8e9e1fe00d53a0b16`.
- Candidate source: `083967e40159b664e59daf40b63f512430f697cb`.
- Epoch: `1789336933000000000`; start2026-09-13T22:04:06Z.
- Prompt SHA256: `4cdd27b78cafc975882f99117d619ce4b971e583b50a60f3653602c92201fed3`.
- Existing MixedQ2 DenseFix GGUF and rank-local affine2 Engram, TP2+EP on two
  gfx1151 EVO-X3 nodes, USB4 Socket, no speculation.
- Temperature0, seed1, context4096, prefix caching OFF. P has35input tokens,
  Q37. Each arm: excluded P32 warmup, Pfirst128, Prepeat128, Qnew128,
  Qrepeat128. Performance requests deliberately ignore EOS; they are not
  presented as completed quality responses.
- One load, fixed A1/B1/B2/A2/A3/B3 order. A uses normal advice, B random.
  Each arm opens fresh embedding file descriptions, clears only the decoded
  LRUs, and discards clean cache for exactly the two identified Engram files
  and their matching read-only mappings. Materialized linears are retained.
  Every interior tensor range must have zero resident pages by `mincore`
  before warmup, on both ranks. All36ranges/rank pass.

Fresh opens, row clearing and scoped file-cache preparation are test-only.
They are not added to production requests. This matches table state between
arms, rather than comparing an A with cold tables against a warmed B. The
first measured P follows a shared32-token warmup; it is not an untouched
post-boot first request. Only table residency, not all system state, is matched.

The frozen gate requires all three new-Q pairs faster on each rank, at least5%
mean wall reduction, median physical-read bytes at most half of A, and each
warm workload's median decode at least98% of A, plus exact correctness.
All gates pass, without threshold changes. The final validator also rejects
wrong preparation scope, mapping counts and file identities. Six injected
setup-invalid cases are rejected; revalidation uses existing raw only.

## Measurements

Rank0 client observations, three samples per row. Decode uses127intervals for
128output tokens. Parentheses contain sample standard deviation, not a
confidence interval. TTFT is engine-reported, not isolated prefill time.

| Workload | Advice | Decode median TPS (SD) | TTFT median s | Request wall median s (SD) | Physical reads median MB |
| --- | --- | ---: | ---: | ---: | ---: |
| Pfirst | normal | 9.6930 (0.0769) | 4.9547 | 18.0557 (0.1104) | 700.195 |
| Pfirst | random | 10.8462 (0.0447) | 4.9464 | 16.6750 (0.0489) | 24.125 |
| Prepeat | normal | 12.6494 (0.0233) | 4.9572 | 14.9934 (0.0206) | 0 |
| Prepeat | random | 12.6686 (0.0279) | 4.9489 | 14.9751 (0.0160) | 0 |
| Qnew | normal | 9.0711 (0.2138) | 5.5919 | 19.5938 (0.3500) | 1058.021 |
| Qnew | random | 10.3397 (0.0745) | 5.3749 | 17.6577 (0.0866) | 39.690 |
| Qrepeat | normal | 12.6410 (0.0853) | 5.1552 | 15.2095 (0.0726) | 0 |
| Qrepeat | random | 12.6619 (0.0301) | 5.1210 | 15.1525 (0.0236) | 0 |

Primary **mean** Qnew wall: rank0 19.421426 ->17.691355s, reduction8.908052%;
rank1 19.415108 ->17.683718s, reduction8.917746%. All three pairs win on
both ranks. Rank0 pairs in seconds:19.593841->17.789692,
19.651754->17.626677,19.018683->17.657698. No slow sample is removed.

Median Qnew physical process reads fall96.249%/96.217% on ranks0/1.
Reader elapsed means, including CPU gather/dequantization and waits:

| Qnew metric | Rank0 A -> B | Rank1 A -> B |
| --- | ---: | ---: |
| Before-first reader seconds | 0.340148 -> 0.147875 | 0.440047 -> 0.243121 |
| Decode reader seconds | 2.268802 -> 1.050274 | 3.065813 -> 1.668184 |
| Total reader seconds | 2.608950 -> 1.198149 | 3.505860 -> 1.911305 |
| Major-fault median | 8344 -> 9690 | 8835 -> 10322 |

Major faults increase while bytes and elapsed time decrease. Smaller physical
reads, rather than fewer fault events, are the benefit. Reader timers are not
pure disk wait, cannot be added across ranks to derive critical-path savings,
and do not explain every remaining millisecond. Physical read counters cover
the whole observed process/request; no conversion from fault counts is used.

HTTP throughput, HTTP TTFT and isolated prefill TPS are **not measured**.
There is no draft acceptance metric in this target-only run. Short fixed
prompts do not establish a universal gain across contexts or workloads.

## Initialization, preparation and memory

Initialization: rank0 328.580s, rank1 328.957s. No init speedup is claimed.
Per-arm test preparation, including barrier and outside generation timings:

| Arm | Rank0 s | Rank1 s |
| --- | ---: | ---: |
| A1 | 0.147916 | 0.170647 |
| B1 | 0.245719 | 0.248235 |
| B2 | 0.155772 | 0.134627 |
| A2 | 0.159367 | 0.135934 |
| A3 | 0.251958 | 0.251947 |
| B3 | 0.250620 | 0.250871 |

Total test preparation1.211352/1.192261s, retained separately. Production
advice makes no page reads or extra row allocation by design; its individual
startup syscall latency is not separately benchmarked. Test mincore vectors
peak at3000108/3000150bytes, below the16MiB bound; masks are capped at64KiB.

Boundary-sampled ranges over all38requests, GiB:

| Accounting | Rank0 | Rank1 |
| --- | ---: | ---: |
| Host MemAvailable | 31.826-32.261 | 33.005-33.224 |
| Process RssAnon | 1.639-1.661 | 1.517-1.542 |
| Process RssFile | 0.236-1.586 | 0.233-1.777 |
| Process VmSwap | 0.3966-0.3986 | 0.5184-0.5203 |

Process RSS is CPU virtual-memory accounting, not total GPU/GTT/model memory.
These are sampled ranges, not continuous peaks. Global swap counters over
request intervals increase by25774/418pages in/out on node01,667/1 on node02;
for the24performance requests,22681/417 and252/0. They include other host
activity, so neither absence of swap activity nor exclusive model attribution
is claimed. The largest node01 swap-in interval is A2-Pfirst,11884pages,
with unchanged process VmSwap. Source-tagged file-reader evidence and matched
output/I/O results support the scoped fix without labeling all faults swap.

Direct observer overhead peaks at0.155288%/0.173680% of request wall, below1%,
and is not subtracted. Maps/smaps and preparation occur outside requests.
Live child PIDs are recorded but their work is not attributed to model phases.

## Correctness and retained negatives

All38request token streams match between ranks. P/Q streams match across
arms/repeats and the independent earlier baseline run024. Warmups equal P's
first32tokens. A/B quality outputs match exactly and stop naturally:

| Task | Output tokens | Result |
| --- | ---: | --- |
| Arithmetic | 2 | 323, PASS |
| Coding | 101 | first_missing_positive,9/9 independent cases PASS |
| JSON | 15 | total17, valid_ids a/c, PASS |
| Reasoning-high | 85 | 100doors final answer10, PASS |

This preserves deterministic outputs within the tested perimeter, not a
general proof of model intelligence. Quality B follows A on warmed tables;
quality timings are therefore not an A/B performance claim.

Attempt025 remains a setup failure: init329.015s, no generation. The harness
looked for embeddings in GC enumeration, but vLLM freezes the model heap.
The correction follows the owned model graph and explicit sidecar references,
with a frozen-GC regression. Candidate math and gates did not change.
The earlier CPU preparation negative also remains: tensor-interior discard
left pages resident on ext4. Exact-file test-only discard corrected setup
before the model attempt; the strict zero-residency gate was never relaxed.

## Deployment and rollback

Launcher default: `DS41_ENGRAM_RANDOM_ADVICE=1`, explicitly forwarded to both
systemd rank units by `pair.sh`. Effective advice is logged once per embedding.
Invalid flag values are rejected before ownership acquisition. CPU/mock tests
cover default forwarding, whole-pair rollback forwarding, invalid values,
duplicate start, stale receipts and unverifiable peer cleanup. Advice tests6/6,
experiment tests4/4, diagnostic tests3/3 pass. No post-result model run.

```sh
cd /home/funboy/StrixHaloClusterDS41
bash runtime/ds41/pair.sh stop
DS41_ENGRAM_RANDOM_ADVICE=0 bash runtime/ds41/pair.sh start "$(date +%s)"
```

Rollback preserves the same source/math/weights and restores normal advice
on both ranks. Stop the whole pair before starting a fresh epoch with1 to
restore the promoted policy. These are operator commands, not an automatic
rerun. An existing running pair is not reconfigured by duplicate start.

Supervised cleanup completed2026-09-13T22:19:02Z: both units/cgroups
OFF_VERIFIED, owner NONE/OFF. GLM remains OFF, gateway available. No benchmark,
model load, new kernel or API/UI work follows this result.

## Evidence index and revalidation

Raw root: `reports/DS41-Q2-001/` in the existing local repository.

- `attempt023/`: closed WO_B candidate, original rank logs/JSON/events.
- `attempt024/`: first diagnostic, frozen prompts, per-rank offline JSON,
  event logs, source/phase fault records and mapping snapshots. Tracked
  [existing-raw audit](attempt024-existing-raw-audit.md) and
  [diagnosis](attempt024-fault-diagnosis.md).
- `attempt025/`: `CPU-PREPARATION-NEGATIVE.md`, CPU logs, preregistration,
  source hashes, rank startup/failure logs and paired cleanup.
  [Setup-abort report](attempt025-setup-abort.md).
- `attempt026/preregister.json`, `prompt-tokens.json`, `source-hashes.txt`,
  `start-once.sh`, `start-receipt.txt`, `pre-start-status.txt`,
  `artifact-preflight-rank0.txt`, `cpu-frozen-heap-tests.log`.
- `attempt026/offline-rank0.json`, `offline-rank0.events.jsonl`, `rank0.log`,
  `engram-preparation-rank0.json`, `fault_maps_*.json`.
- `attempt026/node02/`: corresponding rank1 offline JSON, events, log,
  preparation, mapping snapshots and preflight receipts.
- `attempt026/engram-advice-validation.json`, `supervisor.log`,
  `cleanup-status.txt`, `FINAL.md`; all samples and failure observations retained.
- [Compact result JSON](attempt026-engram-advice-result.json) includes exact
  performance samples/statistics and frozen gates for both ranks.

Revalidate existing data without loading a model:

```sh
python3 scripts/validate-ds41-engram-advice.py \
  --rank0 reports/DS41-Q2-001/attempt026/offline-rank0.json \
  --rank1 reports/DS41-Q2-001/attempt026/node02/offline-rank1.json \
  --reference reports/DS41-Q2-001/attempt024/offline-rank0.json \
  --output /tmp/ds41-attempt026-revalidation.json
```

Next exact action: retain the promoted scoped advice and its paired rollback.
This diagnosis/correction campaign is complete. No automatic additional tuning.
