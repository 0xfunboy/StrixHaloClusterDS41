#!/usr/bin/env bash
set -euo pipefail
ROLE=\${1:?coordinator|worker}
ROOT=/home/funboy/StrixHaloClusterDS41
DS4=$ROOT/.vendor/ds4-v41-halo-7d0454b
MODEL=/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf
CTX=\${DS4_USABLE_CTX:-16384}
COORD=\${DS4_USABLE_COORD:-10.55.0.1}
TP_PORT=\${DS4_USABLE_TP_PORT:-9911}
API_PORT=\${DS4_USABLE_API_PORT:-8080}
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
site=$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')
core="$site/_rocm_sdk_core"; devel="$site/_rocm_sdk_devel"; libs="$site/_rocm_sdk_libraries"; torchlib="$site/torch/lib"
[[ -x "$devel/bin/hipcc" ]] && tool="$devel" || tool="$core"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$core/lib:$tool/lib:$libs/lib:$torchlib"
export OMP_NUM_THREADS=1
export DS4_TP_GATE_TIMEOUT_MS=5000
[[ -f "$MODEL" && $(stat -c %s "$MODEL") == 365713686528 ]] || { echo DS4_Q2_SIZE_GATE_FAIL >&2; exit 20; }
case "$ROLE" in
 coordinator)
   exec 9>/home/funboy/.local/state/strix-cluster/compute.lock
   flock -n 9 || { echo CLUSTER_LIFECYCLE_LOCK_BUSY >&2; exit 75; }
   exec "$DS4/ds4-server" --rocm -m "$MODEL" --ctx "$CTX" \
     --tensor-parallel --role coordinator --listen "$COORD" "$TP_PORT" --transport tcp \
     --batched-session 1 --host 127.0.0.1 --port "$API_PORT"
   ;;
 worker)
   exec "$DS4/ds4" --rocm -m "$MODEL" --ctx "$CTX" \
     --tensor-parallel --role worker --coordinator "$COORD" "$TP_PORT" --transport tcp
   ;;
 *) echo invalid_role >&2; exit 2;;
esac
