#!/usr/bin/env bash
set -Eeuo pipefail
SRC_HOST=${DS41_R4_SRC_HOST:-02-evo-x3-tb}
SRC=${DS41_R4_SRC:-/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf}
DST_DIR=${DS41_R4_DST_DIR:-/home/funboy/models/ds41/ds4-v41-q2}
FILE=${DS41_R4_FILE:-DeepSeek-V4.1-Flash-Q2.gguf}
EXPECT_SIZE=${DS41_R4_SIZE:-365713686528}
EXPECT_SHA=${DS41_R4_SHA:-1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42}
REGISTRY=${DS41_R4_MIRROR_REGISTRY:-/home/funboy/StrixHaloClusterDS41/reports/DS41-Q2-001/recovery-upstream/r4/q2-mirror-registry.json}
LOCK=${DS41_R4_MIRROR_LOCK:-/home/funboy/.local/state/haloclu-ds41/q2-mirror.lock}
PART="$DST_DIR/$FILE.part"
FINAL="$DST_DIR/$FILE"
mkdir -p "$DST_DIR" "$(dirname "$REGISTRY")" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo 'mirror lock busy'; exit 73; }
write_registry() {
  local state=$1 detail=$2 bytes=${3:-0}
  python3 - "$REGISTRY" "$state" "$detail" "$bytes" "$EXPECT_SIZE" "$EXPECT_SHA" "$SRC_HOST" "$SRC" "$PART" "$FINAL" <<'PY'
import json,os,sys,time
p,state,detail,got,total,sha,host,src,part,final=sys.argv[1:]
o={"schema":"ds41-r4-q2-mirror-v1","state":state,"detail":detail,"bytes_received":int(got),"expected_size":int(total),"expected_sha256":sha,"source_host":host,"source_path":src,"part_path":part,"final_path":final,"updated_unix":time.time()}
t=p+'.tmp'; open(t,'w').write(json.dumps(o,indent=2)+'\n'); os.replace(t,p)
PY
}
onerr(){ rc=$?; b=0; [[ -f "$PART" ]] && b=$(stat -c %s "$PART" || echo 0); write_registry FAILED "mirror runner exited rc=$rc; partial preserved" "$b"; exit $rc; }
trap onerr ERR INT TERM
remote_size=$(ssh -o IdentityAgent=none -o BatchMode=yes "$SRC_HOST" "stat -c %s '$SRC'")
[[ "$remote_size" == "$EXPECT_SIZE" ]] || { write_registry FAILED "source size mismatch $remote_size" 0; exit 4; }
existing=0; [[ -f "$PART" ]] && existing=$(stat -c %s "$PART")
if [[ -f "$FINAL" ]]; then
  final_size=$(stat -c %s "$FINAL")
  [[ "$final_size" == "$EXPECT_SIZE" ]] || { write_registry FAILED "unexpected final size $final_size" "$final_size"; exit 5; }
  got=$(ionice -c2 -n7 nice -n19 sha256sum "$FINAL" | awk '{print $1}')
  [[ "$got" == "$EXPECT_SHA" ]] || { write_registry FAILED "existing final sha mismatch $got" "$final_size"; exit 6; }
  write_registry COMPLETE "existing final verified" "$final_size"; exit 0
fi
avail=$(python3 - "$DST_DIR" <<'PY'
import os,sys
s=os.statvfs(sys.argv[1]); print(s.f_bavail*s.f_frsize)
PY
)
remain=$((EXPECT_SIZE-existing)); reserve=$((50*1024*1024*1024))
(( avail >= remain + reserve )) || { write_registry BLOCKED "space gate: available=$avail remaining=$remain reserve=$reserve" "$existing"; exit 7; }
write_registry IN_FLIGHT "resumable USB4 rsync started at low IO/CPU priority" "$existing"
ionice -c2 -n7 nice -n19 rsync -a --partial --append-verify --info=progress2 --rsync-path='ionice -c2 -n7 nice -n19 rsync' -e 'ssh -o IdentityAgent=none -o BatchMode=yes' "$SRC_HOST:$SRC" "$PART"
size=$(stat -c %s "$PART")
[[ "$size" == "$EXPECT_SIZE" ]] || { write_registry FAILED "destination size mismatch $size" "$size"; exit 8; }
write_registry VERIFYING "full destination SHA256 in progress" "$size"
got=$(ionice -c2 -n7 nice -n19 sha256sum "$PART" | awk '{print $1}')
[[ "$got" == "$EXPECT_SHA" ]] || { write_registry FAILED "destination sha mismatch $got" "$size"; exit 9; }
mv "$PART" "$FINAL"
write_registry COMPLETE "mirror and full SHA256 verification complete on NODE01" "$EXPECT_SIZE"
trap - ERR INT TERM
