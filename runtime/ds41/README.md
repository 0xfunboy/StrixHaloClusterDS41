# DS41 experimental runtime overlay

This directory is the reproducible, isolated work for **DS41-Q2-001**. It does
not modify `/home/funboy/StrixHaloClusterGLM` or its `.engine`.

State at this checkpoint: component loader/reader primitives verified. The first
model-bearing load reached DeepSeek V4.1/TP2/Engram initialization but was OOM-killed
before the API because the GGUF iterator cloned mmap-backed tensors and ordinary TP2
would split Q2_K expert blocks. The current overlay removes that anonymous load copy
and uses expert-parallel for complete routed experts; end-to-end inference is still
unqualified until the next target-only run completes.

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

- backbone: `/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2`
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
GLM product. The experiment operator must stop/start GLM through its own
whole-pair lifecycle separately and record the restore receipt.

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
