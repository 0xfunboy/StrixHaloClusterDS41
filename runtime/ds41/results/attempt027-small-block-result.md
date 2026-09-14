M=1: 78.669 ms/token, mediana dei tre controlli contemporanei.
Verifica B=2/4: N/A ms/blocco qualificati. Gate di fedelta fallito prima delle misure.
Correttezza a prefisso comune: FAIL per B2 e B4; M1 conserva il riferimento.
Kernel/fallback dominanti: B2/B4 eseguono MoE Triton e mHC/projection eager; dominanza temporale per operatore non misurata.
Margine disponibile per drafter e gestione: N/A, manca un costo di verifica qualificato.
Decisione: percorso bloccato sulla fedelta M1/M>1. Non preparare ancora DSpark.

# Attempt027: small-block verification feasibility

## Result

The existing V2 target can execute B2/B4 same-sequence speculative verification
after a committed prefix. It does not preserve the promoted M1 reference on
this test. Both ranks reproduce the same failure. No block speed promotion,
DSpark integration, download or target-kernel change was made.

Tested source: `90639eb815ee1eff7d09707fc12d903eff28e5db`.
Parent promoted runtime: `977ffbf6ceb78e54d6f11b395dbd8dbfe438e923`.
One model load, epoch `1789343164000000000`, initialization338.730s on rank0.
Nine requests per rank, completed without a runtime exception. The supervisor
verified both ranks OFF at `2026-09-13T23:59:16Z`; GLM remains OFF and gateway
health returned HTTP200. Historical attempt026 and WO_B closure are unchanged.

## Configuration and actual path

MixedQ2 DenseFix + Engram2, TP2+EP, two gfx1151 EVO-X3, Socket USB4, one sequence,
eager, context4096, prefix caching OFF. Native HIP MoE, M1 coefficient/Sinkhorn,
M1 TileLang projection/RMS and scoped Engram MADV_RANDOM remain enabled; WO_B0.
Pinned vLLM `0bfb653d3b5161660db9ada0d84c2cdd60961de7`, plugin
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`, llama GGUF helpers
`cd628010bc3fc0a787d156c969d52a0789451c96`. Fast artifact identity passes on
both ranks without rehashing unchanged weight payloads.

The35-token P prompt and64 oracle completion tokens come from attempt026.
Prompt specification SHA256:
`bdf0788ff08d22cf581ca2f0582399c5f6a224c430774c6616f35d232ad89ee7`.
Greedy temperature0/seed1; proposals activate after8 emitted tokens. The oracle
is diagnostic replay, not a trained drafter. B is one already-emitted anchor
input plus K drafts; at most B NEW outputs including the bonus, not B+1.

The active pinned V2 runner rejects ordinary ngram/custom_class. The bounded
adapter admits exactly the replay class, supplies known tokens, truncates
scheduler-facing proposal lengths, and bypasses an unused None MTP getter on
the outer model wrapper. Target arithmetic, real rejection, position metadata
and cache/scheduler implementations remain intact. The adapter is imported only
by the explicit offline experiment and is absent from normal API operation.
Config/factory preflight and9CPU bridge tests passed before the model load.

## Correctness before speed

Frozen gates: identical common-prefix input tokens; finite full129280-vocabulary
pre-sampler logits; exact top1; per-position relative L2<=0.005 and max absolute
error<=0.125. No thresholds were changed after results.

| Path | Oracle output | First output difference (1-based) | First eligible logit failure | Relative L2 / max absolute at that position | Block timing |
| --- | --- | ---: | ---: | --- | --- |
| M1 | Exact64/64; all3 measured32-token prefixes exact | None | None | Reference path | 3 clean controls |
| B2 | FAIL | 41 | Position42 | 0.087329907 / 1.9609375 | Not admitted |
| B4 | FAIL | 29 | Position42 | 0.073625567 / 1.634765625 | Not admitted |

Positions are zero-based model positions, not output indices. The first packet
starts at position42 after the same prompt and8 emitted tokens. It already
fails numeric tolerances while top1 still agrees. Frozen validation retains
34 common-prefix eligible rows for B2 and24 for B4: top1 agrees33/34 and23/24,
but zero rows pass all numeric gates. First top1 mismatch is at position74 for
B2, reference1479 versus418, and position62 for B4, reference8205 versus666.
Both ranks have identical output streams across all9requests. An independent
CPU-only raw audit also finds bit-identical cross-rank full logits and exact
B2/B4 versus M1 logits at pre-activation positions34..41. Departure starts at
the first width change, not before it. At B4 position62 the candidate logits
tie tokens8205 and666; deterministic argmax chooses666. This still fails the
unchanged exact-token and numeric gates. Rank agreement does not establish
agreement with M1.

M1's warmup and diagnostic reproduce the historical oracle, and all measured
M1 prefixes match it. The validator's zero-error M1 logit row is a self-reference
check, not a second independently captured historical logit comparison.

After drift, the replay adapter detects a sampled-token mismatch and stops
offering drafts for that request; the target continues normally. It does not
force acceptance or overwrite outputs. Subsequent divergent suffixes are not
valid common-prefix comparisons. Intentional first/last-draft rejection tests
were correctly skipped because clean B4 failed. Reject/partial accept/rollback
and causal-invariance qualification remain N/A, not PASS. The fixture does not
cross the128-token compression boundary or qualify long contexts.

## Contemporary timing and budget

Excluded: one warmup per shape, all diagnostic requests and logits host copies.
Only M1 passed eligibility and received3 clean35-input/32-output requests.
Each timing trial selects24 steady steps after activation. The statistic pairs
the same positions and sums the maximum of the two rank-local complete-step
times. It is not a global HTTP latency measurement.

| M1 trial | Complete step ms/token | Selected-step tokens/s |
| --- | ---: | ---: |
| 1 | 79.724878 | 12.543136 |
| 2 | 78.629450 | 12.717881 |
| 3 | 78.668817 | 12.711517 |
| Median | 78.668817 | 12.711517 |

Sample standard deviation of trial mean ms/token:0.621393. Range78.629450 to
79.724878. Rank0 whole-request medians: wall7.321170s, offline TTFT4.840715s,
first-to-last decode12.488330TPS. Rank1:7.320991s,4.841044s,12.489059TPS.
These short capped diagnostic workloads are not natural-stop quality tests,
HTTP samples or a replacement for attempt020/026 workload baselines.

The timer includes scheduling/metadata, forward, collectives, logits, greedy
rejection, synchronous oracle proposal, post-step and GPU completion. It excludes
pre-step synchronization/counter snapshots and post-timer record assembly.
Whole-request metrics retain surrounding engine/harness work. Replay proposal
batch-wall medians range2.625929 to2.690602ms across rank/trial; they include
tensor synchronization and are not real drafter compute. They are not subtracted.

At the contemporary median, serial target cost is157.337634ms for2tokens and
314.675267ms for4tokens. For a future CORRECT block of cost tB, the favorable
zero-real-drafter/perfect-acceptance bound would be B/tB, and the remaining
additional drafting/rejection/integration budget B*78.668817ms-tB. Here tB is
N/A: failed shapes received no clean performance trials. Therefore the rate,
speedup, remaining budget, real drafter time and real acceptance are all N/A.
Diagnostic wall times retained in raw must not be used to claim speculative TPS.

## Dispatch and resources

Every inspected M1 decode step has40 native MoE calls,80 fused coefficient
calls and80 TileLang projection calls. A real B2/B4 packet instead has40
Triton MoE fallbacks,80 coefficient fallbacks and80 projection fallbacks,
each attributed to `tokens_not_1`. No per-operator duration was captured,
so these counts identify dispatch, not measured time dominance.

Native MoE uses Q8_1 activation quantization for gate/up and down. The EP-aware
Triton fallback uses the original BF16 activations. This is a known mathematical
difference, not merely a generic floating-point explanation. The mHC and
attention paths also change shape. Final logits alone do not locate the first
bad operator or prove MoE is the sole cause. The next diagnosis must isolate it.

Both ranks sampled approximately82.192041GiB allocated and83.109375GiB reserved
by PyTorch. Maximum allocated difference between shape requests is1024bytes;
these are step-boundary snapshots, not transient peaks. Shared Kmax3 buffer
allocation and warmup precede all arms, so this is not incremental memory
versus a separately loaded production M1 runtime. No real draft model exists.
Do not add GPU allocation to RSS as independent physical RAM on unified memory.

Across the entire request phase, including warmups and CPU logits retention:
rank0 major faults+10257, physical reads45,027,328bytes; rank1+10902 and
47,423,488bytes. End process RSS is2,551,576/2,624,936KiB, process swap
742,308/795,448KiB. These counters are not attributed specifically to B2/B4
or clean timing samples. Engram advice and row-cache policy were not retuned.

## Decision and bounded next work

Do not integrate DSpark yet. The prerequisite that failed is faithful target
verification, before a valid block-speed comparison. This is a reproducible
runtime-path blocker, not a physical limitation of the two Strix Halo systems.

The next separately authorized gate should localize the first operator at
position42 on the frozen common prefix, then align M>1 arithmetic with the
promoted M1 path. Start with the explicit MoE activation-quantization difference;
retain routing/global reductions and test other shape-dependent operators if
that counterfactual does not explain the discrepancy. A minimal diagnostic
counterfactual is row-wise reuse of the qualified native M1 routed MoE inside
`GGUFMoEMethod.apply`, after existing routing, with contiguous1xH inputs and
unchanged1x6 routing rows. Preserve expert-map/remote-negative handling, shared
experts and the existing batched TP reduction outside that loop. This is a
localization control, not a speed optimization. Compare matched input activations,
routes and local outputs at the first FFN. mHC runs before the FFN, so inputs
may already differ. Native versus Triton also changes weight reconstruction
and accumulation; an improvement would implicate the numerical path as a whole,
not prove Q8_1 alone responsible. None of this counterfactual was implemented
or run in attempt027. Do not simply loosen the
M1-only guard, extend kernels blindly, or turn off promoted quality checks.
Only after numeric and rejection/state gates pass should block timing decide
between a small-M kernel extension and a real drafter integration.

No second load, alternate engine, new weights, driver changes, automatic kernel
extension or broader campaign follows this result. Existing production defaults
and whole-pair rollback are unchanged.

## Evidence and offline reproduction

Local raw root: `reports/DS41-Q2-001/attempt027/`.

- `prompt-tokens.json`, `preregister.json`, `start-once.sh`: frozen sequence,
  source identity, exact one-shot launch and supervision.
- `source-hashes-rank0.txt`, `identity-rank0.txt`, `peer-preflight.txt`:
  rank source/fast artifact identity and CPU tests.
- `entrypoint-preflight.json`, `bridge-config-final-preflight.json`,
  `bridge-tests-rank0.txt`: config rejection and bounded bridge compatibility.
- `block-verification-rank0.json`, `node02/block-verification-rank1.json`,
  `offline-rank0.json`, `node02/offline-rank1.json`: requests, steps, dispatch,
  scheduler state, proposal events, memory and final gates.
- `diagnostic-{D1,B2,B4}-logits-rank{0,1}.pt`: full captured CPU logits;
  peer files are also retained under `node02/`.
- `rank0.log`, `node02/rank1.log`, `offline-rank0.events.jsonl`,
  `node02/offline-rank1.events.jsonl`: complete run logs and output tokens.
- `validation.json`, `resource-and-timing-summary.json`,
  `summarize-resources.py`: unchanged frozen validator and postrun arithmetic.
- `common-prefix-forensics.py`, `common-prefix-forensics.json`,
  `common-prefix-forensics.md`: independent CPU-only, pre-drift full-logit
  comparisons and source/prompt/rank cross-checks. No thresholds changed.
- `verify-full-rank-logits.py`, `full-rank-logits-equality.json`: separate
  CPU-only equality receipt covering all64/65/67 captured logit rows for
  D1/B2/B4, including non-oracle suffixes; finite and exact across ranks.
- `validator-cpu-tests.py`, `validator-cpu-tests.md`,
  `torch-cpu-validator.txt`: synthetic PASS and11 injected failures rejected.
- `setup-negative.md`, `validation-setup-negative.json`: SDK environment
  setup failures preserved separately. Correcting LD_LIBRARY_PATH for CPU raw
  loading changes neither measured data nor thresholds.
- `start-receipt.txt`, `supervisor.log`, `final-pair-status.txt`:
  both-rank execution identity and verified cleanup.

CPU-only revalidation on node01, no model load:

```sh
cd /home/funboy/StrixHaloClusterDS41
bash reports/DS41-Q2-001/attempt027/revalidate.sh
```

Expected exit1, overallFAIL with B2/B4 correctness failures and M1PASS.
The one-shot launch remains archived for provenance, not an instruction to
repeat an unchanged failed test. Raw tensors are kept locally, not uploaded
as repository weight-like blobs. Compact results and diagnostic source are
published in this repository.
