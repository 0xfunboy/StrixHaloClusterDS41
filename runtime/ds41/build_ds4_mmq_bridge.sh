#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd -P)
PIN=8db1d1d155cb0400a86a86b9c62d0defb3a6148b
SRC=${DS4_SRC:?set DS4_SRC to a clean antirez/ds4 checkout containing $PIN}
OUT=${1:?output .so path}
[[ "$(git -C "$SRC" rev-parse "$PIN^{commit}")" == "$PIN" ]] || { echo 'DS4 pin unavailable' >&2; exit 2; }
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
HIPCC="$VENV/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc"
[[ -x "$HIPCC" ]] || exit 2
TMP=$(mktemp -d /tmp/ds41-ds4-build.XXXXXX)
cleanup(){ git -C "$SRC" worktree remove --force "$TMP/ds4" >/dev/null 2>&1 || true; rm -rf "$TMP"; }
trap cleanup EXIT
git -C "$SRC" worktree add --detach "$TMP/ds4" "$PIN" >/dev/null
git -C "$TMP/ds4" apply "$ROOT/runtime/ds41/ds4_mmq_interleaved_w13.patch"
cd "$TMP/ds4"
FLAGS=(-O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=gfx1151 -std=c++17 -DGGML_USE_HIP -DDS4_HIP_MMQ_Y=64 -Icuda/mmq -fPIC)
for spec in \
 'cuda/mmq/ds4_ggml_stubs.cu:ds4_ggml_stubs.o' \
 'cuda/mmq/ds4_mmq.cu:ds4_mmq.o' \
 'cuda/mmq/quantize.cu:quantize.o' \
 'cuda/mmq/mmid.cu:mmid.o' \
 'cuda/mmq/mmvq.cu:mmvq.o' \
 'cuda/mmq/test/d2r_stubs.cu:d2r_stubs.o'; do
  IFS=: read -r src obj <<<"$spec"
  "$HIPCC" "${FLAGS[@]}" -c "$src" -o "$TMP/$obj"
done
"$HIPCC" "${FLAGS[@]}" -c "$ROOT/runtime/ds41/ds4_prefill_bridge.cu" -o "$TMP/bridge.o"
mkdir -p "$(dirname "$OUT")"
"$HIPCC" -shared -o "$OUT" "$TMP"/{bridge,ds4_ggml_stubs,ds4_mmq,quantize,mmid,mmvq,d2r_stubs}.o -lm -pthread -lhipblas -lhipblaslt -lrocblas
sha256sum "$OUT"
