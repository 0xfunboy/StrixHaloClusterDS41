#!/usr/bin/env bash
set -euo pipefail

SRC=/home/funboy/worktrees/ds4-prefill-gap-002
RAW=/home/funboy/reports/DS4-PREFILL-GAP-002
ART=$RAW/artifacts/diag
VENV=/home/funboy/StrixHaloClusterGLM/.engine/venv
EXPECTED=a8f44737ecc6bbd406d796d1e402b312f00d1564
PEER=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)
SCP=(scp -q -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5)

mkdir -p "$RAW/build-diag" "$ART"
cd "$SRC"
test "$(git rev-parse HEAD)" = "$EXPECTED"
git diff -- ds4_tp.c rocm/ds4_rocm_tp.cuh > "$RAW/build-diag/diagnostic.patch"
test -s "$RAW/build-diag/diagnostic.patch"

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
  hostname
  uname -a
  "$HIPCC" --version | head -5
  git rev-parse HEAD
  git status --short
} > "$RAW/build-diag/environment.txt"

make -j6 strix-halo ROCM_ARCH=gfx1151 HIPCC="$HIPCC"
bash speed-bench/build-rocm-v41-warmup.sh
./ds4-bench-warm --help > "$RAW/build-diag/ds4-bench-warm.help.txt"

install -m 0755 ds4 "$ART/ds4"
install -m 0755 ds4-bench-warm "$ART/ds4-bench-warm"
sha256sum "$ART/ds4" "$ART/ds4-bench-warm" > "$RAW/build-diag/node01.sha256"

"${PEER[@]}" "mkdir -p '$ART'"
"${SCP[@]}" "$ART/ds4" "02-evo-x3-tb:$ART/ds4"
"${PEER[@]}" "chmod 0755 '$ART/ds4'; sha256sum '$ART/ds4'" > "$RAW/build-diag/node02.sha256"

LOCAL=$(sha256sum "$ART/ds4" | awk '{print $1}')
REMOTE=$(awk '{print $1}' "$RAW/build-diag/node02.sha256")
test "$LOCAL" = "$REMOTE"

date -Is > "$RAW/build-diag/BUILD_COMPLETE"
