#!/usr/bin/env bash
set -euo pipefail

ROLE=${1:?worker|server}
TP_PORT=${2:?tp_port}
API_PORT=${3:-18080}
CTX=${4:-69632}
KV_DIR=${5:-}
RELEASE=/home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-engram1
MODEL=/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
COORD=10.55.0.1

SITE=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
CORE="$SITE/_rocm_sdk_core"
DEVEL="$SITE/_rocm_sdk_devel"
LIBS="$SITE/_rocm_sdk_libraries"
TORCHLIB="$SITE/torch/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CORE/lib:$DEVEL/lib:$LIBS/lib:$TORCHLIB"
export OMP_NUM_THREADS=1
export DS4_TP_GATE_TIMEOUT_MS=5000
unset DS4_V41_DISABLE_ENGRAM_CONCURRENT
unset DS4_V41_ENGRAM_TIMING
unset DS4_ROCM_V41_VERIFY2

test -x "$RELEASE/ds4"
test -x "$RELEASE/ds4-server"
test -f "$MODEL"
test "$(stat -c %s "$MODEL")" = "365713686528"

case "$ROLE" in
  worker)
    exec "$RELEASE/ds4" --rocm -m "$MODEL" --ctx "$CTX" \
      --role worker --coordinator "$COORD" "$TP_PORT" \
      --tensor-parallel --transport tcp
    ;;
  server)
    extra=()
    if [[ -n "$KV_DIR" ]]; then
      case "$KV_DIR" in
        /home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001/p4/kv-*) ;;
        *) echo "invalid campaign KV dir: $KV_DIR" >&2; exit 22 ;;
      esac
      mkdir -p "$KV_DIR"
      extra+=(--kv-disk-dir "$KV_DIR" --kv-disk-space-mb 8192)
    fi
    exec "$RELEASE/ds4-server" --rocm -m "$MODEL" --ctx "$CTX" \
      --tensor-parallel --role coordinator --listen "$COORD" "$TP_PORT" \
      --transport tcp --batched-session 1 --host 127.0.0.1 --port "$API_PORT" \
      "${extra[@]}"
    ;;
  *)
    echo "invalid role: $ROLE" >&2
    exit 2
    ;;
esac
