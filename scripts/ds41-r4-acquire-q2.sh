#!/usr/bin/env bash
set -euo pipefail
REPO=${DS41_R4_REPO:-antirez/deepseek-v4.1-flash-gguf}
REV=${DS41_R4_REV:-dd8a266f7145edc19e2334b46e19b6821f221dc7}
FILE=${DS41_R4_FILE:-DeepSeek-V4.1-Flash-Q2.gguf}
EXPECT_SIZE=${DS41_R4_SIZE:-365713686528}
EXPECT_SHA=${DS41_R4_SHA:-1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42}
OUT_DIR=${DS41_R4_OUT_DIR:?set DS41_R4_OUT_DIR}
REGISTRY=${DS41_R4_REGISTRY:?set DS41_R4_REGISTRY}
HF=${DS41_R4_HF:-/home/funboy/StrixHaloClusterGLM/.engine/venv/bin/hf}
mkdir -p "$OUT_DIR" "$(dirname "$REGISTRY")"
write_registry() {
  local state=$1 detail=${2:-}
  STATE="$state" DETAIL="$detail" python3 - "$REGISTRY" "$REPO" "$REV" "$FILE" "$EXPECT_SIZE" "$EXPECT_SHA" "$OUT_DIR" <<'PY'
import json, os, sys, time
p,repo,rev,file,size,sha,out=sys.argv[1:]
obj={"schema":"ds41-r4-q2-acquisition-v1","state":os.environ["STATE"],"detail":os.environ.get("DETAIL", ""),
     "repo":repo,"revision":rev,"file":file,"expected_size":int(size),"expected_sha256":sha,
     "out_dir":out,"updated_unix":time.time()}
tmp=p+".tmp"; open(tmp,"w").write(json.dumps(obj,indent=2)+"\n"); os.replace(tmp,p)
PY
}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_registry FAILED "exit=$rc"; fi' EXIT
write_registry IN_FLIGHT "single Internet copy on NODE02; pinned revision"
"$HF" download "$REPO" "$FILE" --repo-type model --revision "$REV" --local-dir "$OUT_DIR"
python3 - "$OUT_DIR/$FILE" "$EXPECT_SIZE" "$EXPECT_SHA" <<'PY'
import hashlib, os, sys
p,size,sha=sys.argv[1],int(sys.argv[2]),sys.argv[3]
st=os.stat(p)
if st.st_size != size: raise SystemExit(f"size mismatch {st.st_size} != {size}")
h=hashlib.sha256()
with open(p,'rb',buffering=0) as f:
    while True:
        b=f.read(16<<20)
        if not b: break
        h.update(b)
got=h.hexdigest()
if got != sha: raise SystemExit(f"sha256 mismatch {got} != {sha}")
print(f"VERIFIED size={st.st_size} sha256={got}")
PY
write_registry COMPLETE "download and SHA256 verification complete on NODE02"
trap - EXIT
