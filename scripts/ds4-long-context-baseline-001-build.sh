#!/usr/bin/env bash
set -euo pipefail

SRC=/home/funboy/worktrees/ds4-long-context-baseline-001-e1
RAW=/home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001
RELEASE=/home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-engram1
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
EXPECTED=a8f44737ecc6bbd406d796d1e402b312f00d1564

mkdir -p "$RAW/build"
cd "$SRC"
test "$(git rev-parse HEAD)" = "$EXPECTED"
test -z "$(git status --short --untracked-files=no)"

SITE=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
CORE="$SITE/_rocm_sdk_core"
DEVEL="$SITE/_rocm_sdk_devel"
LIBS="$SITE/_rocm_sdk_libraries"
TORCHLIB="$SITE/torch/lib"
HIPCC="$DEVEL/bin/hipcc"
test -x "$HIPCC"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CORE/lib:$DEVEL/lib:$LIBS/lib:$TORCHLIB"
export HIPCC

{
  date -Is
  uname -a
  "$HIPCC" --version | head -5
  git rev-parse HEAD
  git diff --stat 7d0454b4e32ef1e90235f2b001d6643b5934438c..HEAD
} > "$RAW/build/environment.txt"

make -j6 strix-halo ROCM_ARCH=gfx1151 HIPCC="$HIPCC"
bash speed-bench/build-rocm-v41-warmup.sh
./ds4-bench-warm --help > "$RAW/build/ds4-bench-warm.help.txt"

sha256sum ds4 ds4-server ds4-bench ds4-bench-warm "$RELEASE/ds4" "$RELEASE/ds4-server" > "$RAW/build/binaries.sha256"
stat -c '%n %s' ds4 ds4-server ds4-bench ds4-bench-warm "$RELEASE/ds4" "$RELEASE/ds4-server" > "$RAW/build/binaries.stat"
date -Is > "$RAW/build/BUILD_COMPLETE"
