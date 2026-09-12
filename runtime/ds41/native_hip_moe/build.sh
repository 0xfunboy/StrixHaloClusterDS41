#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/funboy/StrixHaloClusterDS41
VENDOR="$ROOT/.vendor/gguf-plugin"
PIN=d4c1f0d082fc7cd4350da56689109a01c1f29d6c
PATCH="$ROOT/runtime/ds41/native_hip_moe/moe-negskip.patch"
SRC=/home/funboy/.cache/ds41-native-hip-moe-src
BUILD=/home/funboy/.cache/ds41-native-hip-moe-build
BT="$BUILD/temp"
BL="$BUILD/lib"
DEST=/home/funboy/models/ds41/native-hip-moe
OUT="$DEST/_C_gguf.abi3.so"
MANIFEST="$DEST/build-manifest.json"
ENGINE=/home/funboy/StrixHaloClusterGLM/.engine
VENV="$ENGINE/venv"
TARGET_REL=vllm_gguf_plugin/csrc/gguf/moe_vec.cuh
HIP_REL=vllm_gguf_plugin/csrc/gguf/moe_vec_hip.cuh
OLD='if (row >= nrows) {'
NEW='if (expert < 0 || row >= nrows) {'

[[ "$(git -C "$VENDOR" rev-parse HEAD)" == "$PIN" ]] || {
  echo "plugin pin mismatch" >&2; exit 2;
}
[[ -x "$VENV/bin/python" && -f "$PATCH" ]] || exit 2

cleanup() {
  set +e
  if git -C "$VENDOR" worktree list --porcelain 2>/dev/null | grep -Fqx "worktree $SRC"; then
    git -C "$VENDOR" worktree remove --force "$SRC" >/dev/null 2>&1 || true
  fi
  rm -rf "$SRC" "$BUILD"
}
trap cleanup EXIT
cleanup
mkdir -p "$(dirname "$SRC")" "$BT" "$BL" "$DEST"
git -C "$VENDOR" worktree prune
git -C "$VENDOR" worktree add --detach "$SRC" "$PIN"

# Use an exact one-occurrence replacement rather than relying on patch fuzz.
"$VENV/bin/python" - "$SRC/$TARGET_REL" "$OLD" "$NEW" <<'PY'
import pathlib, sys
path=pathlib.Path(sys.argv[1]); old=sys.argv[2]; new=sys.argv[3]
text=path.read_text()
count=text.count(old)
if count != 1:
    raise SystemExit(f"expected exactly one guard to replace, found {count}")
path.write_text(text.replace(old,new,1))
PY

grep -Fq "$NEW" "$SRC/$TARGET_REL" || {
  echo "native HIP source guard missing after replacement" >&2; exit 3;
}
source_guard_count=$(grep -Fc "$NEW" "$SRC/$TARGET_REL")
[[ "$source_guard_count" == 1 ]] || { echo "unexpected source guard count=$source_guard_count" >&2; exit 3; }
source_sha=$(sha256sum "$SRC/$TARGET_REL" | awk '{print $1}')

site=$($VENV/bin/python - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)
core="$site/_rocm_sdk_core"
devel="$site/_rocm_sdk_devel"
libs="$site/_rocm_sdk_libraries"
torchlib="$site/torch/lib"
tool="$devel"
export PATH="$tool/bin:$tool/lib/llvm/bin:$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$tool/lib:$core/lib:$libs/lib:$torchlib"
export ROCM_PATH="$tool" ROCM_HOME="$tool" HIP_PATH="$tool"
export HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"
export PYTORCH_ROCM_ARCH=gfx1151
export MAX_JOBS="${MAX_JOBS:-2}"

cd "$SRC"
"$VENV/bin/python" setup.py build_ext --build-temp "$BT" --build-lib "$BL"

# PyTorch hipify emits a sibling *_hip.cuh consumed by gguf_kernel.hip.  The
# build is invalid unless the remote-expert guard survived that transformation.
[[ -f "$SRC/$HIP_REL" ]] || { echo "hipified moe_vec source missing" >&2; exit 4; }
grep -Fq "$NEW" "$SRC/$HIP_REL" || {
  echo "native HIP guard missing from hipified source" >&2; exit 4;
}
hip_guard_count=$(grep -Fc "$NEW" "$SRC/$HIP_REL")
[[ "$hip_guard_count" == 1 ]] || { echo "unexpected hip guard count=$hip_guard_count" >&2; exit 4; }
# setup/hipify must not have silently restored the original CUDA source either.
grep -Fq "$NEW" "$SRC/$TARGET_REL" || {
  echo "native HIP guard disappeared from CUDA source during build" >&2; exit 4;
}
hip_sha=$(sha256sum "$SRC/$HIP_REL" | awk '{print $1}')

so=$(find "$BL" -type f -name '_C_gguf*.so' -print -quit)
[[ -n "$so" && -f "$so" ]] || { echo 'native HIP library missing' >&2; exit 5; }
install -m 0755 "$so" "$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"

sha=$(sha256sum "$OUT" | awk '{print $1}')
patch_sha=$(sha256sum "$PATCH" | awk '{print $1}')
host=$(hostname)
torch_info=$($VENV/bin/python - <<'PY'
import json, torch
print(json.dumps({"torch":torch.__version__,"hip":torch.version.hip}))
PY
)

"$VENV/bin/python" - "$OUT" "$MANIFEST" "$sha" "$patch_sha" "$PIN" "$host" "$torch_info" "$source_sha" "$hip_sha" <<'PY'
import json, pathlib, sys, torch
so, manifest, sha, patch_sha, pin, host, torch_info, source_sha, hip_sha = sys.argv[1:]
torch.ops.load_library(so)
for op in ("ggml_moe_a8_vec","ggml_mul_mat_vec_a8","ggml_moe_get_block_size"):
    if not hasattr(torch.ops._C_gguf, op):
        raise SystemExit(f"missing op {op}")
obj={
  "schema":"ds41-native-hip-moe-build-v2",
  "host":host,
  "arch":"gfx1151",
  "plugin_pin":pin,
  "patch_sha256":patch_sha,
  "source_guard_verified":True,
  "hipified_guard_verified":True,
  "patched_source_sha256":source_sha,
  "hipified_source_sha256":hip_sha,
  "library_path":so,
  "library_sha256":sha,
  "torch":json.loads(torch_info),
  "ops":{
    "ggml_moe_a8_vec":str(torch.ops._C_gguf.ggml_moe_a8_vec.default._schema),
    "ggml_mul_mat_vec_a8":str(torch.ops._C_gguf.ggml_mul_mat_vec_a8.default._schema),
  },
}
pathlib.Path(manifest).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
print(json.dumps(obj,indent=2,sort_keys=True))
PY

echo "DS41_NATIVE_HIP_BUILD_PASS sha256=$sha source_sha=$source_sha hip_sha=$hip_sha path=$OUT manifest=$MANIFEST"
