#!/usr/bin/env bash
set -euo pipefail
SRC_HOST=${DS41_R4_SRC_HOST:-02-evo-x3-tb}
SRC=${DS41_R4_SRC:-/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf}
DST_DIR=${DS41_R4_DST_DIR:-/home/funboy/models/ds41/ds4-v41-q2}
FILE=${DS41_R4_FILE:-DeepSeek-V4.1-Flash-Q2.gguf}
EXPECT_SIZE=${DS41_R4_SIZE:-365713686528}
EXPECT_SHA=${DS41_R4_SHA:-1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42}
REGISTRY=${DS41_R4_MIRROR_REGISTRY:?set DS41_R4_MIRROR_REGISTRY}
mkdir -p "$DST_DIR" "$(dirname "$REGISTRY")"
command -v rsync >/dev/null || exit 3
remote_size=$(ssh "$SRC_HOST" "stat -c %s '$SRC'")
[ "$remote_size" = "$EXPECT_SIZE" ] || exit 4
part="$DST_DIR/$FILE.part"
rsync -a --partial --append-verify --info=progress2 "$SRC_HOST:$SRC" "$part"
python3 - "$part" "$EXPECT_SIZE" "$EXPECT_SHA" <<'PY'
import hashlib,os,sys
p,size,sha=sys.argv[1],int(sys.argv[2]),sys.argv[3]
if os.stat(p).st_size != size: raise SystemExit('size mismatch')
h=hashlib.sha256()
with open(p,'rb',buffering=0) as f:
    for b in iter(lambda:f.read(16<<20),b''): h.update(b)
g=h.hexdigest()
if g != sha: raise SystemExit(f'sha mismatch {g}')
print(g)
PY
mv "$part" "$DST_DIR/$FILE"
