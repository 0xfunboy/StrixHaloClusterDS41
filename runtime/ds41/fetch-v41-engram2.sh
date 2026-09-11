#!/usr/bin/env bash
set -Eeuo pipefail
repo='Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP'
rev='802f1a00982705d81b79ad1c83aa0ccc0b863ebc'
dest=${1:-/home/funboy/models/ds41/engram2-source}
reserve=$((32*1024*1024*1024))
mkdir -p "$dest"
files=(
  'model-00047-of-00048.safetensors|30769728248|e80b0d1481bb8e32196398d5dd670df721604517ecdfdf279485cc92ca6a19d2'
  'model-00048-of-00048.safetensors|30770569376|1aee6ffb59d1314d3264c95d61109ec18df61aa48101073b45f1c6730e8d9230'
)
need=0
for row in "${files[@]}"; do IFS='|' read -r name size sha <<<"$row"; [[ -f "$dest/$name" ]] || need=$((need+size)); done
avail=$(df -B1 --output=avail "$dest" | tail -1 | tr -d ' ')
(( avail >= need + reserve )) || { echo "insufficient disk: avail=$avail need=$need reserve=$reserve" >&2; exit 2; }
for row in "${files[@]}"; do
  IFS='|' read -r name size sha <<<"$row"
  out="$dest/$name"; part="$out.part"
  if [[ -f "$out" ]]; then
    actual=$(stat -c %s "$out")
    [[ "$actual" == "$size" ]] || { echo "$name final size mismatch" >&2; exit 3; }
    echo "$sha  $out" | sha256sum -c -
    continue
  fi
  url="https://huggingface.co/$repo/resolve/$rev/$name"
  echo "DS41_FETCH_START $name size=$size revision=$rev"
  curl --fail --location --continue-at - --retry 6 --retry-delay 2 --retry-all-errors \
       --output "$part" "$url"
  actual=$(stat -c %s "$part")
  [[ "$actual" == "$size" ]] || { echo "$name partial size $actual != $size" >&2; exit 4; }
  echo "$sha  $part" | sha256sum -c -
  mv "$part" "$out"
  sync -f "$out" 2>/dev/null || true
  echo "DS41_FETCH_COMPLETE $name sha256=$sha"
done
printf '{"repo":"%s","revision":"%s","status":"COMPLETE","files":2}\n' "$repo" "$rev" > "$dest/receipt.json"
