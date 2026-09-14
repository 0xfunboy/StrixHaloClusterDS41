#!/usr/bin/env bash
set -Eeuo pipefail
repo='deepseek-ai/DeepSeek-V4.1-Flash'
rev='2bc89ac599031fa673cab993f1df02fc4a98c673'
dest=${1:-/home/funboy/models/ds41/dspark-v41-mtp-2bc89ac}
reserve=$((8*1024*1024*1024))
mkdir -p "$dest"
files=(
  'model-00044-of-00048.safetensors|2652728736|9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05'
  'model-00045-of-00048.safetensors|2573998176|0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5'
  'model-00046-of-00048.safetensors|2706402896|e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d'
)
need=0
for row in "${files[@]}"; do
  IFS='|' read -r name size sha <<<"$row"
  if [[ -f "$dest/$name" ]]; then
    actual=$(stat -c %s "$dest/$name")
    [[ "$actual" == "$size" ]] || { echo "$name existing final size mismatch: $actual != $size" >&2; exit 3; }
    echo "$sha  $dest/$name" | sha256sum -c - >/dev/null
  else
    part="$dest/$name.part"
    cur=0; [[ -f "$part" ]] && cur=$(stat -c %s "$part")
    (( cur <= size )) || { echo "$name partial oversized: $cur > $size" >&2; exit 3; }
    need=$((need + size - cur))
  fi
done
avail=$(df -B1 --output=avail "$dest" | tail -1 | tr -d ' ')
(( avail >= need + reserve )) || { echo "insufficient disk: avail=$avail need=$need reserve=$reserve" >&2; exit 2; }
for row in "${files[@]}"; do
  IFS='|' read -r name size sha <<<"$row"
  out="$dest/$name"; part="$out.part"
  if [[ -f "$out" ]]; then
    echo "DS41_DSPARK_FETCH_ALREADY_VERIFIED $name size=$size sha256=$sha"
    continue
  fi
  url="https://huggingface.co/$repo/resolve/$rev/$name"
  echo "DS41_DSPARK_FETCH_START $name size=$size revision=$rev current=$(stat -c %s "$part" 2>/dev/null || echo 0)"
  curl --fail --location --continue-at - --retry 12 --retry-delay 2 --retry-all-errors \
       --output "$part" "$url"
  actual=$(stat -c %s "$part")
  [[ "$actual" == "$size" ]] || { echo "$name partial size $actual != $size" >&2; exit 4; }
  echo "$sha  $part" | sha256sum -c -
  mv "$part" "$out"
  sync -f "$out" 2>/dev/null || true
  echo "DS41_DSPARK_FETCH_COMPLETE $name size=$size sha256=$sha"
done
python3 - "$dest" "$repo" "$rev" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); repo=sys.argv[2]; rev=sys.argv[3]
files=[]
for p in sorted(root.glob('model-0004[4-6]-of-00048.safetensors')):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
receipt={'schema':'ds41-dspark-mtp-fetch-v1','repo':repo,'revision':rev,
         'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'status':'COMPLETE','files':files,'total_bytes':sum(x['bytes'] for x in files)}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY
