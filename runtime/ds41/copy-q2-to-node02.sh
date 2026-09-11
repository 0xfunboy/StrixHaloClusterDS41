#!/usr/bin/env bash
set -Eeuo pipefail
src=/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2/
dest=/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2/
ssh=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5)
[[ $(hostname) == 01-EVO-X3 ]]
[[ $("${ssh[@]}" 02-evo-x3-tb hostname) == 02-EVO-X3 ]]
"${ssh[@]}" 02-evo-x3-tb 'mkdir -p /home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2'
rsync -a --partial --human-readable --info=progress2 \
  -e 'ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5' \
  "$src" 02-evo-x3-tb:"$dest"
"${ssh[@]}" 02-evo-x3-tb "cd '$dest' && sha256sum -c SHA256SUMS"
echo DS41_Q2_NODE02_COPY_COMPLETE
