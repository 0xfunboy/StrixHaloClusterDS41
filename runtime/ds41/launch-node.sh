#!/usr/bin/env bash
set -Eeuo pipefail
if (( $# != 4 )); then
  echo "usage: $0 NODE_RANK HOST_IP EPOCH API_PORT" >&2; exit 2
fi
rank=$1 host_ip=$2 epoch=$3 api_port=$4
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
[[ "$(hostname)" == "0$((rank+1))-EVO-X3" ]] || { echo 'host/rank mismatch' >&2; exit 2; }
[[ "$epoch" =~ ^[0-9]{10,20}$ && "$api_port" =~ ^[0-9]{4,5}$ ]] || exit 2
ROOT=${DS41_ROOT:-/home/funboy/StrixHaloClusterDS41}
ENGINE=/home/funboy/StrixHaloClusterGLM/.engine
VENV="$ENGINE/venv"
VLLM_SOURCE="$ROOT/.vendor/vllm-dsv41"
PLUGIN_SOURCE="$ROOT/.vendor/gguf-plugin"
GGUF_PY="$ROOT/.vendor/llama-v41/gguf-py"
ARTIFACT_CONFIG="$ROOT/runtime/ds41/artifact.json"
MODEL_DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_dir"])' "$ARTIFACT_CONFIG")
MODEL_FILE="$MODEL_DIR/DSV41-mixedq2-00001-of-00005.gguf"
ENGRAM_DIR="/home/funboy/models/ds41/engram2-tp2/rank$rank"
for p in "$VENV/bin/python" "$VLLM_SOURCE/vllm/__init__.py" "$PLUGIN_SOURCE/vllm_gguf_plugin/__init__.py" "$GGUF_PY/gguf/__init__.py" "$MODEL_FILE" "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json" "$ENGRAM_DIR/model-00047-of-00048.safetensors" "$ENGRAM_DIR/model-00048-of-00048.safetensors"; do
  [[ -e "$p" ]] || { echo "missing DS41 runtime input: $p" >&2; exit 2; }
done
site=$($VENV/bin/python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
core="$site/_rocm_sdk_core"; devel="$site/_rocm_sdk_devel"; libs="$site/_rocm_sdk_libraries"; torchlib="$site/torch/lib"
if [[ -x "$devel/bin/hipcc" ]]; then tool="$devel"; else tool="$core"; fi
if [[ -x "$tool/lib/llvm/bin/clang" ]]; then llvm="$tool/lib/llvm/bin"; else llvm="$tool/llvm/bin"; fi
mkdir -p "$ROOT/.cache/rank$rank/aiter" "$ROOT/.cache/rank$rank/triton" "$ROOT/.cache/rank$rank/torchinductor"
export PATH="$tool/bin:$llvm:$VENV/bin:/usr/local/bin:/usr/bin:/bin"
# Prefer the runtime SDK path consistently in parent and registry subprocesses.
# Core/devel copies can be hardlinks but rocprofiler compares registration paths.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$core/lib:$tool/lib:$libs/lib:$torchlib"
export PYTHONPATH="$ROOT:$VLLM_SOURCE:$PLUGIN_SOURCE:$GGUF_PY:$core/share/amd_smi"
export ROCM_PATH="$tool" ROCM_HOME="$tool" HIP_PATH="$tool" HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"
export CMAKE_PREFIX_PATH="$tool/lib/cmake:$torchlib/../share/cmake"
export XDG_CACHE_HOME="$ROOT/.cache/rank$rank" AITER_JIT_DIR="$ROOT/.cache/rank$rank/aiter"
export TRITON_CACHE_DIR="$ROOT/.cache/rank$rank/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/rank$rank/torchinductor"
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 PYTORCH_ROCM_ARCH=gfx1151 VLLM_TARGET_DEVICE=rocm
unset CUDA_VISIBLE_DEVICES
export VLLM_HOST_IP="$host_ip" VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=0 VLLM_ROCM_USE_SKINNY_GEMM=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_GGUF_USE_CUDA=0
export DS41_LOAD_PHASE_LOG=1 DS41_DROP_SHARD_CACHE=1 DS41_STREAM_TEXT_WEIGHTS=1 DS41_MOE_C_LEGACY_TEXT_ABI=1 DS41_DECOMPOSED_QKV_INSERT=1
export DS41_EP_SKIP_REMOTE="${DS41_EP_SKIP_REMOTE:-1}"
export DS41_NATIVE_HIP_MOE="${DS41_NATIVE_HIP_MOE:-1}"
export DS41_MHC_COEFF_SINKHORN="${DS41_MHC_COEFF_SINKHORN:-1}"
export DS41_MHC_PROJECTION_RMS="${DS41_MHC_PROJECTION_RMS:-1}"
export DS41_ENGRAM_RANDOM_ADVICE="${DS41_ENGRAM_RANDOM_ADVICE:-1}"
[[ "$DS41_ENGRAM_RANDOM_ADVICE" == 0 || "$DS41_ENGRAM_RANDOM_ADVICE" == 1 ]] || { echo 'invalid DS41_ENGRAM_RANDOM_ADVICE' >&2; exit 2; }
export DS41_ENGRAM2_DIR="$ENGRAM_DIR" DS41_ENGRAM_CACHE_ROWS="${DS41_ENGRAM_CACHE_ROWS:-65536}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=1 OMP_NUM_THREADS=1
export DS41_PREFILL_TELEMETRY="${DS41_PREFILL_TELEMETRY:-0}"
export DS41_REPEAT_DIAG_DIR="${DS41_REPEAT_DIAG_DIR:-}"
export DS41_REPEAT_DIAG_PROMPT_TOKENS="${DS41_REPEAT_DIAG_PROMPT_TOKENS:-1588}"
export DS41_REPEAT_DIAG_MAX_REQUESTS="${DS41_REPEAT_DIAG_MAX_REQUESTS:-2}"
export NCCL_SOCKET_IFNAME='=thunderbolt0' GLOO_SOCKET_IFNAME=thunderbolt0 NCCL_NET=Socket NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_SOCKET_NTHREADS=1 NCCL_NSOCKS_PERTHREAD=1 NCCL_DEBUG=WARN
export MASTER_ADDR=10.55.0.1 MASTER_PORT="${DS41_MASTER_PORT:-29741}"
cd "$ROOT"
"$VENV/bin/python" -m runtime.ds41.artifact_identity verify-fast --rank "$rank"
run_mode=${DS41_RUN_MODE:-api}
[[ "${DS41_MOE_PREFILL_BLOCK_M:-4}" == 4 || "${DS41_MOE_PREFILL_BLOCK_M:-4}" == 8 ]] || { echo 'invalid DS41_MOE_PREFILL_BLOCK_M' >&2; exit 2; }
export DS41_MOE_PREFILL_BLOCK_M="${DS41_MOE_PREFILL_BLOCK_M:-4}"
serving_preset=${DS41_SERVING_PRESET:-}
dspark_k=${DS41_REAL_DSPARK_K:-}
if [[ -n "$serving_preset" ]]; then
  [[ "$run_mode" == api && "$serving_preset" == dspark-k2-gfx1151 && "$dspark_k" == 2 ]] || { echo 'invalid DS41 serving preset/K'; exit 2; }
  export DS41_DSPARK_MXFP4_BF16=1 DS41_NATIVE_HIP_MOE_ROWWISE=1 DS41_MHC_ROWWISE_BLOCK=1 DS41_ATTN_WOB_LLMM1=0
  [[ -d /home/funboy/models/ds41/dspark-v41-mtp-2bc89ac ]] || { echo 'missing DSpark sidecar'; exit 2; }
fi
case "$run_mode" in
  offline)
    exec "$VENV/bin/python" -m torch.distributed.run \
      --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
      --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
      -m runtime.ds41.offline_spmd
    ;;
  api)
    api_args=(
      --model "$MODEL_FILE" --hf-config-path "$MODEL_DIR" --tokenizer "$MODEL_DIR"
      --served-model-name DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2
      --host "$host_ip" --port "$api_port" --api-server-count 1
      --tensor-parallel-size 2 --pipeline-parallel-size 1 --enable-expert-parallel
      --distributed-executor-backend external_launcher --language-model-only
      --config-format gguf --load-format gguf --quantization gguf --dtype bfloat16
      --attention-backend ROCM_FLASHMLA_SPARSE_DSV4
      --max-model-len "${DS41_API_MAX_MODEL_LEN:-65664}" --block-size 128 --max-num-seqs 1 --max-num-batched-tokens 1024
      --kv-cache-memory-bytes 1073741824 --kv-cache-dtype auto
      --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling
      --enforce-eager --seed 1 --generation-config vllm --enable-per-request-metrics
      --reasoning-parser deepseek_v41
    )
    if [[ -n "$serving_preset" ]]; then
      spec_json='{"method":"dspark","model":"/home/funboy/models/ds41/dspark-v41-mtp-2bc89ac","num_speculative_tokens":2,"quantization":"fp8","enable_adaptive_verification":false,"draft_tensor_parallel_size":2,"draft_load_config":{"load_format":"safetensors","safetensors_load_strategy":"lazy"},"draft_sample_method":"greedy","rejection_sample_method":"standard"}'
      api_args+=(--speculative-config "$spec_json" --per-request-spec-decode-metrics detailed)
    fi
    exec "$VENV/bin/python" -m torch.distributed.run \
      --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
      --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
      -m runtime.ds41.api_server "${api_args[@]}"
    ;;
  *)
    echo "invalid DS41_RUN_MODE=$run_mode" >&2
    exit 2
    ;;
esac
