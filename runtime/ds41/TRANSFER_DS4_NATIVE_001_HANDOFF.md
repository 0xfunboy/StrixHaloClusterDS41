# TRANSFER DS4 → NATIVE 001 — handoff

updated_at: 2026-09-19T03:16+02:00
phase: L0 COMPLETE / L1 FAIL / L2 M1 STARTUP NEGATIVE 002 / PROGRESSIVE CACHE FIX PREP
next_action: finish model-free gate for progressive per-consumed-tensor GGUF cache eviction, freeze a NEW isolated L2 release, then allow one same-configuration startup retry. Do not dispatch quality until rank0/rank1/paired HTTP200 and live release/env identity PASS.
repo_worktree: /home/funboy/worktrees/ds41-transfer-ds4-native-001
repo_branch: exp/ds41-transfer-ds4-native-001
repo_head: 2ba0b7534e2e4fa328925bc1243f162148aadd41 (pushed; progressive-cache delta pending)

## Live resident runtime at this checkpoint

- DS4 owner: `DS4_DOCUMENT_PROFILE_002_20260918`
- DS4 state: READY, coordinator active, worker active, API HTTP200
- DS4 backend/model: DS4 / DeepSeek-V4.1-Flash-Q2
- DS4 target-only: true; DSpark: false
- Native DS41: OFF, owner NONE/OFF, rank0/rank1 OFF, paired backend OFF
- Old K2: OFF
- No TRANSFER model-bearing systemd unit is active.
- Do not perturb DS4 while L2 release/software preparation is incomplete.

## Qualified fallback

DS4 DOCUMENT PROFILE 002 is terminal PASS:
- document panel 6/6 PASS
- soak 24/24 PASS
- real soak elapsed 8185.734 s
- finalizer QUALIFIED_LEFT_READY
- qualified document scope: document-low / DS4_THINK_LOW, max observed qualified input 2039 tokens
- controller: /home/funboy/StrixHaloClusterDS41/scripts/ds4-document-controller.sh
- Q2 path on both nodes:
  /home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf
- verified size: 365713686528
- frozen SHA256 receipt:
  1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42
- no repeat DS4 soak; no automatic restore of historical K2

## L0 result — COMPLETE

Commit: 4d06da930451897dcf037deb8be8990412c5c807

`ds4-low-v1` is opt-in and preserves thinking ON while suppressing only the
numeric renderer-generated Reasoning Effort prefix used by the pinned vLLM LOW.

Verified:
- actual DeepseekV4Renderer full token-ID parity with DS4: 7/7 PASS
- historical NONE/LOW/HIGH/MAX behavior unchanged: 5/5 PASS
- contradictory/invalid profile combinations fail closed: 6/6 PASS
- target and DSpark consume one canonical serialization: PASS
- fresh patch generation: 27/27 byte-identical
- no count-minus-6 workaround

## L1 result — TERMINAL FAIL

Terminal commit: 11c802775988d9a099afd62526bd057e935d3250
Candidate:
- existing DenseFix target + Engram2
- MMQ prefill ON
- canonical-prefill ON
- DSpark K2 real, three-stage path retained
- `ds4-low-v1`, thinking ON, cap total 2048, temp0, seed1
- prefix cache OFF

Exactly 2/6 main requests were used:
1. code2k-v2: INCOMPLETE_NO_FINAL
   - HTTP200
   - prompt 1629 tokens
   - completion 2048/2048 reasoning tokens
   - final chars 0
   - finish_reason=length
   - prefill 21.547 s / 75.60 tok/s
   - wall 126.99 s
   - K2 draft acceptance 86.72% (1299/1498)
2. docs2k-v2: INCOMPLETE_NO_FINAL
   - HTTP200
   - prompt 1306 tokens
   - completion 2048/2048 reasoning tokens
   - final chars 0
   - finish_reason=length
   - prefill 17.901 s / 72.96 tok/s
   - wall 121.86 s
   - K2 draft acceptance 86.60% (1299/1500)

Holdouts and confirms were not sent. L1 is not to be replayed.
Decision per mandate: L1 FAIL -> L2 Antirez Q2 + native Engram, target M1 first.

## L2 recovered/prepared evidence

Recovered from the prior chat:
- `runtime/ds41/transfer-ds4-native-001/l2/antirez-header-inventory.json`
- Antirez GGUF tensor_count: 1046
- no model request or L2 model load was already in flight

Current CPU-only L2 gate on NODE01:
- result: PASS
- raw:
  `runtime/ds41/transfer-ds4-native-001/l2/cpu-gate-node01.raw`
- normalized result:
  `runtime/ds41/transfer-ds4-native-001/l2/cpu-gate-node01.json`
- stderr:
  `runtime/ds41/transfer-ds4-native-001/l2/cpu-gate-node01.stderr`

Verified contracts:
- target mapping: 1038 ordinary target tensors + 8 native Engram tensors = 1046/1046 accounted
- Antirez llama.cpp `.weight/.bias` normalization is sufficient for target names
- no unknown target tensor is silently skipped
- native Engram encoding: e4m3_e8m0_32_row264
- Engram layers: 1, 14
- Engram rows: 384006168, 384016682
- token map exact parity: 129280 entries
- compressed vocab exact: 99092
- compressed pad exact: 2
- primes exact
- hash multipliers exact
- row264 decoder bit-exact with independent scalar DS4 formula on real rows
- non-uniform E8M0 scale bytes exercised on both Engram layers

Antirez native Engram tensor contracts:
- q norm: F32, data shape (4, 5120)
- k norm: F32, data shape (4, 5120)
- WKV: F16, data shape (25600, 6144)
- table: I8 row264, mmap-backed; no whole-table expansion

MMQ compatibility:
- Antirez expert gate/up tensor type is IQ2_XXS (GGML type 16)
- Antirez expert down tensor type is Q2_K (GGML type 10)
- existing DS4 MMQ admission contract explicitly requires weight_type=16 and weight_type2=10
- therefore existing MMQ path is format-compatible for admitted pure target prefill; fallback remains declared elsewhere

## L2 software — mapper fix pushed; progressive-cache startup fix pending freeze

- `runtime/ds41/native_antirez_engram.py`
  - bounded LRU over mmap-backed native GGUF rows
  - exact DS4 E4M3/E8M0 decode + BF16 RNE
  - native q/k/WKV source from same Antirez GGUF
  - no Engram2 mixing
- `runtime/ds41/antirez_artifact_identity.py`
  - read-only fast identity gate using the already-frozen both-node SHA receipt + live size/header contract
  - no full rehash
- `runtime/ds41/gguf_stream_cache.py`
  - opt-in page-aligned `MADV_DONTNEED` + `POSIX_FADV_DONTNEED` for one already-consumed tensor
  - never unmaps or mutates checkpoint bytes; no global cache operation
- `runtime/ds41/transfer-ds4-native-001/l2/deepseek_v41_antirez_adapter.py`
  - fail-closed 1046-tensor accounting
- `scripts/patch-ds41-transfer-native-l2.py`
  - isolated release overlay only
- `scripts/test-ds41-transfer-native-l2-cpu.py`
- `scripts/transfer-ds4-native-001-l2-controller.sh`
  - exact release/attempt ownership checks
  - whole-pair ON/OFF
  - fallback target is qualified DS4, not old K2
- `scripts/run-ds41-transfer-native-l2-m1.py`
  - max six requests, same frozen document order/gates
  - persistent registry and replay guard
- `scripts/finalize-ds41-transfer-native-l2-m1.sh`
  - PASS leaves owned M1 READY
  - FAIL returns whole pair to qualified DS4 and waits for READY

## L2 release contract to build next

Planned isolated release:
`/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001`

Base release:
`native-ds4low-k2-transfer001`

L2 M1 release properties:
- target weights: verified Antirez calibrated Q2
- config/tokenizer source: existing local V4.1 config/tokenizer used by L0 parity
- Engram: native GGUF, not Engram2
- DSpark: OFF for M1
- MMQ prefill: ON only under existing type/shape admission contract
- canonical-prefill: ON
- TP2 / EP2 / BLOCK_M4 retained
- prompt profile: ds4-low-v1
- total output cap: 2048
- temperature: 0
- seed: 1
- no download, no re-quantization, no DenseFix application to Antirez Q2
- RuntimeMax: infinity for the session
- model API max context for this phase: 16384

## L2 M1 release preflight — PASS before lifecycle

Release materialized on both nodes:
`/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001`

Verified before any DS4 OFF:
- release `launch-node.sh`: bash syntax PASS on NODE01 and NODE02
- L2 Python overlay: py_compile PASS on NODE01 and NODE02
- Antirez fast identity rank0 PASS and rank1 PASS
- both identities bind the frozen 365713686528-byte Q2 and SHA receipt `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`
- NODE01 full CPU L2 contract remains PASS: 1038+8 tensor accounting, exact hash metadata/token map, bit-exact row264 decode
- rsync created a new NODE02 release; no existing release was overwritten
- launcher hash observed on both nodes after sync: `488886315e485a8ea5bfc7f05ab6d320c9144c0474ee51765668b970d97d3d65`

Preflight caught and fixed before load one generator defect: the first generated launcher left `fi DS41_ENGRAM_CACHE_ROWS=...` on one line. Current release is bash-valid on both nodes; the patch generator now emits the correct newline/export. No model was loaded during this correction.

## L2 M1 startup negative 001 — LOCALIZED, ZERO REQUESTS

First M1 startup terminalized during GGUF `load_weights`, before READY and before any model request.
- epoch: `1789761472122549252`
- rank0 InvocationID: `0e707e12c9454ed6a02ab50ea9c5ba86`
- rank1 InvocationID: `dad9f3d54e16417eb9a9d6ddbb1d847d`
- rank0 and rank1 exact error: `DS41 streaming loader only accepts the single language_model group, got 'head.weight_type'`
- pre-failure Antirez identity PASS; native Engram runtime hash contract PASS
- both failed DS41 units/cgroups verified OFF; stale owner reconciled with the release `pair.sh reconcile`
- qualified DS4 restored READY HTTP200; K2 OFF
- requests sent: 0

Cause is localized: packed top-level GGUF tensors emit `*.weight_type` companions, while the V4.1 outer streaming mapper rerooted `head.weight`/`embed.weight` but not their companions. The fix adds only:
- `head.weight_type -> language_model.lm_head.weight_type`
- `embed.weight_type -> language_model.model.embed_tokens.weight_type` after prefix+suffix mapping

Both-node post-fix CPU gates PASS 8/8 name-map cases, 1038+8=1046 tensor accounting and the existing real-row native Engram decode. Result: `runtime/ds41/results/transfer-ds4-native-001-l2-startup-negative-001.{json,md}`. One retry of the same M1 configuration is admitted because this is a model-free localized integration fault and no request was sent.

## L2 M1 startup negative 002 — LOCALIZED UMA/FILE-CACHE PRESSURE, ZERO REQUESTS

The same-config retry passed the `weight_type` boundary and advanced into MoE parameter materialization. NODE02 then terminalized with `torch.OutOfMemoryError` while requesting 710 MiB:
- epoch: `1789761988271104620`
- rank0 InvocationID: `2d6a594ff8124a84aa029b18ef6cfbbd`
- rank1 InvocationID: `dd4432be158e45ddb491d355786de6a1`
- reported device/UMA capacity: 120 GiB; free 4.28 MiB
- PyTorch allocated: 62.13 GiB; reserved but unallocated: 76.10 MiB
- requests sent: `0/6`
- Antirez identity PASS; native Engram runtime hash contract PASS; prior `head.weight_type` failure did not recur

The single Antirez GGUF is 365713686528 bytes and the existing loader only drops its clean mmap pages at whole-shard completion. Header-only sizing shows the two ~94.4 GiB native Engram tables are excluded from the ordinary target iterator, while the largest ordinary target tensor is ~1.384 GiB. The next causal fix is therefore progressive clean-page eviction immediately after each yielded target tensor has been synchronously consumed. No re-quantization, checkpoint split, global `drop_caches`, swap manipulation or weight change.

After NODE02 failed, rank0 remained active under the same owner receipt. The whole-pair lifecycle stopped it and returned `DS41_OFF_VERIFIED`; qualified DS4 DOCUMENT PROFILE 002 is restored READY/HTTP200 and K2 remains OFF. Evidence: `runtime/ds41/results/transfer-ds4-native-001-l2-startup-negative-002.{json,md}`.

## Active job

None. DS4 is the resident qualified fallback. The progressive-cache code is model-free work only; no model-bearing process may start until its release preflight is complete.

## Persistence

PLAN needs the startup-negative002/progressive-cache update before the next switch.
Git mapper fix/evidence is pushed at `2ba0b7534e2e4fa328925bc1243f162148aadd41`; progressive-cache helper/generator/test/report are pending scoped commit/push.
Four inherited mode-bit changes and unrelated untracked L0 artifacts remain untouched.
