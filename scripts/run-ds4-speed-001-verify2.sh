#!/usr/bin/env bash
set -Eeuo pipefail

ARM=${1:?serial|batch}
TAG=${2:?unique-tag}
MAX=${3:-520}
[[ "$ARM" == serial || "$ARM" == batch ]] || { echo invalid-arm >&2; exit 2; }
[[ "$MAX" =~ ^[0-9]+$ && "$MAX" -ge 514 && "$MAX" -le 768 ]] || { echo invalid-max >&2; exit 2; }

ROOT=/home/funboy/StrixHaloClusterDS41
RELEASE=/home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-verify2
MODEL=/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf
PROMPT=$ROOT/runtime/ds41/document-profile-002/prompts/code2k-middle-explicit-v2.txt
OUT=/home/funboy/reports/DS4-SPEED-001/e2/$TAG
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
PEER=(ssh -o IdentityAgent=none -o BatchMode=yes 02-evo-x3-tb)
PORT=9921

[[ ! -e "$OUT" ]] || { echo "replay guard: $OUT" >&2; exit 3; }
mkdir -p "$OUT/logits"

site=$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')
ld="/usr/lib/x86_64-linux-gnu:$site/_rocm_sdk_core/lib:$site/_rocm_sdk_devel/lib:$site/_rocm_sdk_libraries/lib:$site/torch/lib"
common="LD_LIBRARY_PATH=$ld OMP_NUM_THREADS=1 DS4_TP_GATE_TIMEOUT_MS=5000"
if [[ "$ARM" == batch ]]; then common="$common DS4_ROCM_V41_VERIFY2=1"; fi

cleanup() {
    systemctl --user stop ds4-speed-001-e2-coordinator.service 2>/dev/null || true
    "${PEER[@]}" 'systemctl --user stop ds4-speed-001-e2-worker.service 2>/dev/null || true' || true
}
trap cleanup EXIT

cleanup
"${PEER[@]}" "mkdir -p '$OUT'"
"${PEER[@]}" "systemd-run --user --unit=ds4-speed-001-e2-worker --collect --property=KillMode=control-group --property=Restart=no --property=TimeoutStopSec=30 --property='StandardOutput=append:$OUT/worker.log' --property='StandardError=append:$OUT/worker.log' env $common '$RELEASE/ds4' --rocm -m '$MODEL' --ctx 1024 --tensor-parallel --role worker --coordinator 10.55.0.1 '$PORT' --transport tcp" >"$OUT/worker-start.txt"

exec 9>/home/funboy/.local/state/strix-cluster/compute.lock
flock -n 9 || { echo CLUSTER_LIFECYCLE_LOCK_BUSY >&2; exit 75; }
set +e
env $common "$RELEASE/ds4-bench" --rocm -m "$MODEL" \
    --tensor-parallel --role coordinator --listen 10.55.0.1 "$PORT" --transport tcp \
    --prompt-file "$PROMPT" --ctx-start 512 --ctx-max "$MAX" --ctx-alloc 1024 \
    --step-incr 2 --gen-tokens 0 --dump-frontier-logits-dir "$OUT/logits" \
    --csv "$OUT/result.csv" >"$OUT/coordinator.stdout" 2>"$OUT/coordinator.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$OUT/exit-code.txt"
exit "$rc"
