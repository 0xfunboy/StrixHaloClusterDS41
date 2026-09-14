# DS41 experimental runtime overlay

This directory is the reproducible, isolated work for **DS41-Q2-001**. It does
not modify `/home/funboy/StrixHaloClusterGLM` or its `.engine`.

Current qualified state: the repaired DenseFix artifact runs DeepSeek V4.1 with
TP2 dense/attention, EP2 routed experts and rank-local Engram2 on the two EVO-X3.
Attempt015 qualified the gfx1151 native HIP M=1 routed path (IQ2_XXS gate/up,
Q2_K down, Q8_1 activations). Attempt018 qualified decode-only mHC coefficient /
softmax / Sinkhorn fusion for the exact M=1, hc_mult=4, 20-iteration V4.1 delayed
pre-mix path. Attempt020 then qualified the pinned TileLang mHC projection/RMS
path for M=1 with K=5120/20480, FP32 weights and accumulation, leaving all other
shapes on the Torch fallback. On the frozen 128-token AB/BA/AB attempt020,
contemporary A averaged 9.54034 tok/s and projection/RMS B averaged 12.08665
tok/s (+26.69% decode); all three paired comparisons favored B. Arithmetic,
coding (9/9 independent tests), JSON and reasoning-high 100-doors passed, and
rank0/rank1 token streams matched for every request. Native HIP, mHC coefficient
fusion and TileLang projection/RMS remain independently rollbackable. Set
`DS41_MHC_PROJECTION_RMS=0` to roll back only projection/RMS, set
`DS41_MHC_COEFF_SINKHORN=0` to roll back coefficient/Sinkhorn fusion, or set
`DS41_NATIVE_HIP_MOE=0` to roll routed MoE back to Triton skip-remote. Raw
attempt020 artifacts remain under `reports/DS41-Q2-001/attempt020/`; tracked
compact results are under `runtime/ds41/results/`.

### WO_B M=1 LLMM1 result (attempt022/023)

**Terminal: correctness PASS, speed gate FAIL, no promotion.** The frozen
attempt023 A/B measured11.70417 tok/s for A and11.89974 tok/s for B (+1.67%).
All three pairs favored B, but the preregistered5% threshold was not met.
Arithmetic, coding (one function, nine independent cases), JSON and
reasoning-high completed naturally and passed. All12 request token streams
matched between ranks; A and B themselves differed from completion index28.
The promoted attempt020 reference remains12.08665 tok/s. Full result:
[`attempt023-wob-ab.json`](results/attempt023-wob-ab.json).

Early measured requests had thousands of major faults on both hosts;
later speed requests had0-2. These counters cover whole requests and do not
localize the source to Engram, weights, swap or decode. All samples are retained.
Neither this narrow test suite nor rank agreement establishes general quality
equivalence. Both DS41 ranks were verified OFF after the run; GLM was unchanged.

Attempt022 generated no tokens: model inspection aborted on conflicting
ROCm profiler registration paths. The scoped launcher fix uses the core SDK
library path before its devel hardlinks. Small import reproducers passed on
both hosts, without driver, package or weight changes. Attempt023 then reused
the original numerical gates and exact measurement protocol.

`DS41_ATTN_WOB_LLMM1=1` opts into a local BF16 output-projection matmul
for input `[1,4096]` and local weight `[5120,4096]` only. It is disabled by
default and is not a promoted preset. The existing TP2 all-reduce still runs
exactly once. All unqualified contracts fall back to the original layer.
The shape guard can include one-token prefill, not just decode; larger prefill
and verification matrices retain the original path. This does not change
WO_A, RoPE, attention, MoE, Engram, weight formats or the Socket transport.

Attempt023 reuses attempt020's tokenized prompts and same-load sequence:
excluded warmup32 for each arm, then A1/B1/B2/A2/A3/B3 at128 output tokens.
A retains all promoted numerical paths; B adds only WO_B LLMM1. Arithmetic,
coding, JSON and reasoning-high tasks run afterward on the same load.
Component tolerances remain rel-L2<=0.005 and max-abs<=0.125. Promotion also
requires coherent ranks, naturally completed passing tasks, all three paired
B>A, and at least5% mean decode gain. Microkernel timings are not model TPS.
Component and setup evidence: `reports/DS41-Q2-001/attempt022/`.
Model A/B raw and final report: `reports/DS41-Q2-001/attempt023/`.

The attempt021 attention fixture containers have zero samples. Component
gates use canonical layer0 intermediates reconstructed from the verified
DenseFix weights, as recorded in their raw provenance. These gates do not
represent all layers or long-context model quality.

## Source pins

See `DEPENDENCIES.lock`. The experiment intentionally reuses the already
qualified ROCm 10 / PyTorch 2.13 / gfx1151 toolchain read-only. The vLLM V4.1
source and GGUF plugin are separate pinned checkouts.

## Recreate source overlays

```bash
git clone https://github.com/vllm-project/vllm.git .vendor/vllm-dsv41
git -C .vendor/vllm-dsv41 checkout 0bfb653d3b5161660db9ada0d84c2cdd60961de7
git -C .vendor/vllm-dsv41 apply runtime/ds41/patches/vllm-dsv41-engram2.patch

git clone https://github.com/vllm-project/vllm-gguf-plugin.git .vendor/gguf-plugin
git -C .vendor/gguf-plugin checkout d4c1f0d082fc7cd4350da56689109a01c1f29d6c
git -C .vendor/gguf-plugin apply runtime/ds41/patches/vllm-gguf-plugin-deepseek-v41.patch

git clone https://github.com/ggml-org/llama.cpp.git .vendor/llama-v41
git -C .vendor/llama-v41 checkout cd628010bc3fc0a787d156c969d52a0789451c96
```

`llama.cpp/gguf-py` is used as the pinned GGUF parser/quant-table source. The
experiment does not claim llama.cpp itself can execute V4.1.

## Verified small tests

```bash
ENGINE_PY=/home/funboy/StrixHaloClusterGLM/.engine/venv/bin/python
PYTHONPATH="$PWD:$PWD/.vendor/vllm-dsv41:$PWD/.vendor/gguf-plugin:$PWD/.vendor/llama-v41/gguf-py" \
  "$ENGINE_PY" scripts/test-ds41-affine-safetensors.py

# Requires the same read-only ROCm runtime environment used for the V4.1 source import.
PYTHONPATH="$PWD:$PWD/.vendor/vllm-dsv41:$PWD/.vendor/gguf-plugin:$PWD/.vendor/llama-v41/gguf-py" \
  "$ENGINE_PY" scripts/test-ds41-disk-engram.py
```

The GGUF adapter is fail-closed and was checked against all five MixedQ2 shards:
1038/1038 raw tensors mapped, zero missing and zero duplicate target names.
Real model bytes were also used to execute IQ2_XXS and Q2_K Triton kernels on
Radeon 8060S/gfx1151 before any model load. After the first load OOM, two additional
fail-closed tests were added: the >1 GiB routed tensor must remain an mmap-backed
CPU view (RssAnon delta <128 MiB), and full-vs-EP2 routed output on real IQ2_XXS/Q2_K
expert bytes must match for both M=1 decode and M=65 MMQ/prefill paths.

## Model files

Weights are deliberately outside Git. Canonical paths used by the experiment:

- runtime backbone (verified DenseFix): `/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix`
- immutable original/recovery source: `/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2`
- source Engram2: `/home/funboy/models/ds41/engram2-source`
- rank-local Engram2: `/home/funboy/models/ds41/engram2-tp2/rank{0,1}`

`fetch-v41-engram2.sh` resumes only incomplete pinned files and verifies the
published SHA-256 before publication. `partition-ds41-engram2.py` creates
rank-local SafeTensors with explicit source SHA and row-range metadata.

## Target-only two-node launcher

`runtime/ds41/launch-node.sh` reuses the qualified ROCm 10 / PyTorch 2.13
runtime read-only and the proven external-launcher topology over `thunderbolt0`.
It is intentionally target-only: no speculative config is passed. Initial
qualification is fixed to 4K context, 1024-token prefill chunks, block size 128,
1 GiB explicit KV cache and eager execution. Tensor parallel remains 2 for dense
attention, while `--enable-expert-parallel` stores complete routed experts 192+192
instead of splitting the Q2_K 2304-wide intermediate at an invalid 1152 boundary.
`DS41_ENGRAM2_DIR` selects the rank-local compact affine2 Engram sidecar.

The orchestration helper is `runtime/ds41/pair.sh`; it never stops or starts the
GLM product. Cluster exclusion is control-plane state, not a lock kept alive by
a model process: `/home/funboy/.local/state/strix-cluster/compute.lock` only
serializes lifecycle mutations, while `owner.json` persists owner/state/epoch,
per-rank nonce and InvocationID. `ACTIVE`, `OFF_VERIFIED` and `UNKNOWN` are
distinct; SSH/systemd/probe failure is always `UNKNOWN`. A stale or
`UNRECONCILED` receipt blocks new starts until an explicit `reconcile` proves
both DS41 units/cgroups OFF.

Each experimental transient rank is bounded locally by systemd
`RuntimeMaxSec=5400`, `TimeoutStopSec=30` and `KillMode=control-group`; this
limits an orphan even when NODE01 later loses SSH, but expiry by itself never
changes the persistent owner receipt to OFF. Stop is peer-first and verified;
NODE01 is not stopped/released while NODE02 is `UNKNOWN`. The experiment
operator controls GLM through its separate lifecycle.

`preflight-ds41-config.py` constructs `EngineArgs/VllmConfig` without loading the
model and fails unless the runtime resolves DeepSeek V4.1, GGUF, TP2+EP, block
128, Engram layers 1/14 and no speculative config.

Additional DS41-Q2-001 preflight commands:

```bash
$ENGINE_PY scripts/test-ds41-gguf-zero-copy.py
$ENGINE_PY scripts/test-ds41-gguf-ep-real.py
```

The first uses the real `blk.0.ffn_gate_exps` tensor and detects an eager anonymous
clone. The second compares a four-expert full reference against the sum of two
expert-parallel shards on the same real quantized bytes. EP mapping never renormalizes
router weights: remote routes are zeroed on each rank and the normal TP all-reduce
reconstructs the global routed contribution.

## Request fault diagnosis

Attempt023 closes WO_B LLMM1 with `CORRECTNESS_PASS / SPEED_GATE_FAIL /
NO_PROMOTION`. The qualified numerical baseline remains `facf864`, with native
HIP MoE, mHC coefficient/Sinkhorn and projection/RMS enabled, WO_B disabled.
The core-first ROCm library lookup fix is retained.

[Existing raw audit](results/attempt024-existing-raw-audit.md) separates same-arm
first-request timing from whole-request fault counters. Attempt024 adds an
opt-in, CPU-only observer to the existing offline runner: excluded warmup32,
P/P/Q/Q/P128, then one uninstrumented P128 overhead control. It observes model
processes at request/first-token/final-token boundaries and tags mapped Engram
reader work by source and phase. No cache resets, additional GPU synchronization
or mathematical changes are involved. Reader elapsed time includes gather and
dequantization, so it is not presented as pure disk wait.

```sh
/home/funboy/StrixHaloClusterGLM/.engine/venv/bin/python scripts/test-ds41-fault-diagnostics.py
```

Diagnostic raw and the frozen prompt file live under
`reports/DS41-Q2-001/attempt024/` locally.

Attempt024 localized approximately99% of request major faults to the two
sparse Engram table readers. The cost recurs on a new prompt and disappears
on its identical repeat. See the [diagnosis](results/attempt024-fault-diagnosis.md).

Attempt026 qualifies `DS41_ENGRAM_RANDOM_ADVICE=1` as the launcher default. It applies
`MADV_RANDOM` only to complete pages contained in each embedding's packed
weight, scale and bias tensors. It adds no row storage, changes no model
arithmetic, and preserves the existing65536-row caches. Unsupported advice
falls back to the normal path with explicit status. Initialization logs
`DS41_ENGRAM_ADVICE` with the effective policy and affected ranges.

The experiment reopens only embedding readers at each arm boundary, retains
materialized linears, clears decoded LRUs, and discards clean cache only for
the two identified Engram files. A zero-residency check on full table pages is mandatory. This
test-only setup is not part of normal inference. AB/BA/AB compares new-request
latency and physical I/O, with separate first-continuation and warm-decode
metrics. Exact output/rank/task agreement is required. Raw and preregistration:
`reports/DS41-Q2-001/attempt026/`. Attempt025's frozen-GC discovery abort
remains recorded separately; it produced no inference sample.

[Attempt026 result](results/attempt026-engram-advice-result.md): new-Q request
wall time falls8.91%, physical reads fall96.25%, and warm decode stays near
12.66TPS. All38rank streams match, with exact A/B output and natural-stop task
checks. These are offline measurements on the fixed35/37-input,128-output
workloads, not HTTP or sustained warm-decode gains.

The controller forwards the advice setting to both transient ranks. With the
pair already running, stop the whole pair before changing the setting:

```sh
bash runtime/ds41/pair.sh stop
# New numeric owner epoch; rollback changes advice only, not weights or kernels.
DS41_ENGRAM_RANDOM_ADVICE=0 bash runtime/ds41/pair.sh start "$(date +%s)"
```

To restore the promoted setting, stop the whole pair and start a fresh epoch
with `DS41_ENGRAM_RANDOM_ADVICE=1` (also the default). An already-running pair
is not reconfigured by a duplicate start. Never restart one rank in isolation.
The final experiment leaves DS41 and GLM OFF, with the existing gateway available.

## Small-block verification gate

[Attempt027](results/attempt027-small-block-result.md) exercises the actual V2
target verification path with known continuation tokens, not a learned drafter.
B2 and B4 both fail common-prefix numerical fidelity against the promoted M1
path. Their speed and DSpark headroom are unqualified. Contemporary M1 steady
step median is78.669ms/token on this short diagnostic workload. All production
optimizations remain intact, and the offline-only replay adapter is not enabled
by the API launcher. Next work requires a separate numerical alignment gate
before a drafter integration or small-M kernel extension.

## Shared cluster ownership

`/home/funboy/.local/state/strix-cluster/compute.lock` serializes lifecycle
mutations, not the model's entire lifetime. The persistent v2 `owner.json`
receipt records owner, epoch, per-rank nonces and systemd InvocationIDs. Starts
require agreement between that receipt and both unit/cgroup probes; a stale
receipt or an unknown peer blocks a new start even when the lock is free.
DS41 starts rank0 first, verifies its identity, then starts rank1. Stop is
peer-first: an unverifiable NODE02 leaves rank0 alive and marks the pair
unreconciled. Only verified removal of both ranks permits `NONE/OFF`. GLM's
existing ownership-aware lifecycle remains separate and unchanged.

## Attempt004 loader diagnosis

Attempt003 proved the zero-copy/EP memory fix but remained before API readiness
for more than three hours with no new I/O/fault progress and three CPU-bound
threads. The root cause was then isolated in upstream V4.1 EP weight loading:
`ExpertMapManager.map_global_to_local()` called `.item()` on a device-resident
384-entry expert map once per expert checkpoint record. About 46k records turned
that scalar GPU->CPU synchronization into hours of serialized loading.

The DS41 vLLM overlay now keeps a tiny CPU mirror used only by scalar checkpoint
mapping. Runtime routing/kernels retain the original device map. The gfx1151 test
`test-ds41-expert-map-cpu.py` verifies both maps and performs 46,080 rank-local
lookups in well under two seconds. `DS41_LOAD_PHASE_LOG=1` also adds bounded
phase/tensor markers so future load stalls can be localized without ptrace,
signals, or kernel-policy changes.
