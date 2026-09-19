#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
VLLM_SOURCE="$ROOT/.vendor/vllm-dsv41"
PLUGIN_SOURCE="$ROOT/.vendor/gguf-plugin"
GGUF_PY="$ROOT/.vendor/llama-v41/gguf-py"
SITE=$("$VENV/bin/python" - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
CORE="$SITE/_rocm_sdk_core"
DEVEL="$SITE/_rocm_sdk_devel"
LIBS="$SITE/_rocm_sdk_libraries"
TORCHLIB="$SITE/torch/lib"
if [[ -x "$DEVEL/bin/hipcc" ]]; then TOOL="$DEVEL"; else TOOL="$CORE"; fi
if [[ -x "$TOOL/lib/llvm/bin/clang" ]]; then LLVM="$TOOL/lib/llvm/bin"; else LLVM="$TOOL/llvm/bin"; fi
CACHE="$ROOT/.cache/attention-parity-002-device"
mkdir -p "$CACHE/aiter" "$CACHE/triton" "$CACHE/torchinductor"
export PATH="$TOOL/bin:$LLVM:$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CORE/lib:$TOOL/lib:$LIBS/lib:$TORCHLIB"
export PYTHONPATH="$ROOT:$VLLM_SOURCE:$PLUGIN_SOURCE:$GGUF_PY:$CORE/share/amd_smi"
export ROCM_PATH="$TOOL" ROCM_HOME="$TOOL" HIP_PATH="$TOOL" HIP_DEVICE_LIB_PATH="$CORE/lib/llvm/amdgcn/bitcode"
export XDG_CACHE_HOME="$CACHE" AITER_JIT_DIR="$CACHE/aiter"
export TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1151 VLLM_TARGET_DEVICE=rocm
unset CUDA_VISIBLE_DEVICES
export VLLM_ROCM_USE_AITER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DS41_V41_ATTN_PARITY=1
exec "$VENV/bin/python" "$ROOT/scripts/test-ds41-v41-attention-parity-002-device.py"
