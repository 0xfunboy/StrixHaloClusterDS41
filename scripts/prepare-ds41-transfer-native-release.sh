#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/home/funboy/worktrees/ds41-transfer-ds4-native-001}
BASE=/home/funboy/.local/share/haloclu-ds41/releases/k2-mmq-engram-9c13117
DEST=/home/funboy/.local/share/haloclu-ds41/releases/native-ds4low-k2-transfer001
PEER=02-evo-x3-tb
commit=$(git -C "$ROOT" rev-parse HEAD)
build_one() {
  local dest=$1
  if [[ -e "$dest" ]]; then
    [[ -f "$dest/runtime/ds41/transfer-ds4-native-001-release.json" ]] || { echo "refuse existing non-transfer release $dest" >&2; return 2; }
    return 0
  fi
  local tmp="${dest}.tmp.$$"
  rm -rf "$tmp"
  cp -a --reflink=auto "$BASE" "$tmp"
  install -m 0644 "$ROOT/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41.py" "$tmp/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41.py"
  install -m 0644 "$ROOT/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41_encoding.py" "$tmp/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41_encoding.py"
  printf '%s\n' "$commit" > "$tmp/.source-commit"
  python3 - "$tmp" "$commit" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); commit=sys.argv[2]
p=root/'runtime/ds41/serving-k2-release.json'
o=json.load(open(p))
o['name']='ds41-native-ds4low-k2-transfer001'
o['qualified_evidence']='TRANSFER_DS4_NATIVE_001_L1_PENDING'
o['profile_status']='EXPERIMENTAL_TRANSFER_CANDIDATE'
o['source_commit']=commit
o['prompt_profile']='ds4-low-v1'
o['numerics']['dspark_k']=2
p.write_text(json.dumps(o,indent=2)+'\n')
rec={
 'schema':'ds41-transfer-ds4-native-001-release-v1',
 'source_commit':commit,
 'base_release':'k2-mmq-engram-9c13117',
 'base_source':'9c13117f56fd81d03c8c610a5fb0148ccdc603b9',
 'prompt_profile':'ds4-low-v1',
 'target':'DenseFix + Engram2',
 'dspark_k':2,
 'mmq_prefill':True,
 'canonical_prefill':True,
 'engram_workers':4,
 'engram_parallel_min_rows':256,
 'ced':False,
}
(root/'runtime/ds41/transfer-ds4-native-001-release.json').write_text(json.dumps(rec,indent=2)+'\n')
PY
  python3 - "$tmp/runtime/ds41/serve-controller.sh" <<'PY'
from pathlib import Path
p=Path(__import__('sys').argv[1])
s=p.read_text()
old='  DS41_ENGRAM_RANDOM_ADVICE=1 DS41_PREFILL_TELEMETRY=1 DS41_API_MAX_MODEL_LEN=65664 \\\n  bash "$ROOT/runtime/ds41/pair.sh" start "$epoch" "$RANK_PORT"'
new='  DS41_ENGRAM_RANDOM_ADVICE=1 DS41_ENGRAM_READ_WORKERS=4 DS41_ENGRAM_PARALLEL_MIN_ROWS=256 \\\n  DS41_PREFILL_TELEMETRY=1 DS41_API_MAX_MODEL_LEN=65664 DS41_CANONICAL_PREFILL_TOPK=1 \\\n  DS41_DS4_MMQ_PREFILL=1 DS41_DS4_MMQ_MIN_TOKENS=128 DS41_DS4_MMQ_MAX_TOKENS=1024 \\\n  bash "$ROOT/runtime/ds41/pair.sh" start "$epoch" "$RANK_PORT"'
if old not in s: raise SystemExit('controller env anchor missing')
p.write_text(s.replace(old,new))
PY
  mv "$tmp" "$dest"
}
build_one "$DEST"
rsync -a --delete --exclude='.cache/' -e 'ssh -o IdentityAgent=none -o BatchMode=yes' "$DEST/" "$PEER:$DEST/"
for host in local peer; do
  if [[ "$host" == local ]]; then cmd=(bash -lc); prefix=''; else cmd=(ssh -o IdentityAgent=none -o BatchMode=yes "$PEER" bash -lc); prefix=''; fi
done
sha_local=$(sha256sum "$DEST/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41.py" "$DEST/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41_encoding.py")
sha_peer=$(ssh -o IdentityAgent=none -o BatchMode=yes "$PEER" "sha256sum '$DEST/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41.py' '$DEST/.vendor/vllm-dsv41/vllm/tokenizers/deepseek_v41_encoding.py'")
[[ "$sha_local" == "$sha_peer" ]] || { echo 'release tokenizer hash mismatch across nodes' >&2; exit 3; }
[[ -f "$DEST/runtime/ds41/lib/libds41-ds4-moe.so" ]] || { echo 'missing MMQ lib' >&2; exit 4; }
ssh -o IdentityAgent=none -o BatchMode=yes "$PEER" "test -f '$DEST/runtime/ds41/lib/libds41-ds4-moe.so'"
echo "DS41_TRANSFER_RELEASE_READY $DEST source=$commit"
