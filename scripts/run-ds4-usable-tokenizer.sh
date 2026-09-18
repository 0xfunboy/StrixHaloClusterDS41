#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
site=$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')
core="$site/_rocm_sdk_core"; devel="$site/_rocm_sdk_devel"; libs="$site/_rocm_sdk_libraries"; torchlib="$site/torch/lib"
[[ -x "$devel/bin/hipcc" ]] && tool="$devel" || tool="$core"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$core/lib:$tool/lib:$libs/lib:$torchlib"
export PYTHONPATH="$ROOT:$ROOT/.vendor/vllm-dsv41:$core/share/amd_smi"
export OMP_NUM_THREADS=1
exec "$VENV/bin/python" "$ROOT/scripts/serve-ds4-usable-tokenizer.py"
