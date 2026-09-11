#!/usr/bin/env bash
set -Eeuo pipefail
root=/home/funboy/models/ds41/engram2-tp2
src="$root/rank1/"; dest="$root/rank1/"; receipt="$root/partition-receipt.json"
ssh_cmd=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5)
[[ $(hostname) == 01-EVO-X3 && $("${ssh_cmd[@]}" 02-evo-x3-tb hostname) == 02-EVO-X3 ]]
[[ -f "$receipt" ]]
"${ssh_cmd[@]}" 02-evo-x3-tb "mkdir -p '$dest'"
rsync -a --partial --human-readable --info=progress2 -e 'ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5' "$src" 02-evo-x3-tb:"$dest"
for layer in 1 14; do
  case "$layer" in 1) name=model-00047-of-00048.safetensors;; 14) name=model-00048-of-00048.safetensors;; esac
  sha=$(jq -r --arg l "$layer" '.layers[$l]["1"].sha256' "$receipt")
  "${ssh_cmd[@]}" 02-evo-x3-tb "echo '$sha  $dest$name' | sha256sum -c -"
done
rsync -a -e 'ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5' "$receipt" 02-evo-x3-tb:"$root/partition-receipt.json"
echo DS41_ENGRAM_RANK1_NODE02_COPY_COMPLETE
