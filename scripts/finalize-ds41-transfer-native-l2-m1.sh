#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/funboy/worktrees/ds41-transfer-ds4-native-001
TERM="$ROOT/reports/DS41-Q2-001/transfer-ds4-native-001/l2-m1/terminal.json"
OUT="$ROOT/reports/DS41-Q2-001/transfer-ds4-native-001/l2-m1/finalizer.json"
CTL="$ROOT/scripts/transfer-ds4-native-001-l2-controller.sh"
DS4CTL=/home/funboy/StrixHaloClusterDS41/scripts/ds4-document-controller.sh

[[ -f "$TERM" ]] || { echo "missing L2 M1 terminal" >&2; exit 2; }
verdict=$(jq -r .status "$TERM")
requests=$(jq -r .requests_used "$TERM")
native=$(bash "$CTL" status | jq -c .native)

atomic_json() {
  local tmp="$OUT.tmp.$$"
  cat >"$tmp"
  mv "$tmp" "$OUT"
}

if [[ "$verdict" == L2_M1_PASS ]]; then
  nstate=$(jq -r .state <<<"$native")
  [[ "$nstate" == READY ]] || {
    echo "L2 M1 PASS terminal but candidate is not READY: $nstate" >&2
    exit 3
  }
  jq -n     --arg status "L2_M1_PASS_LEFT_READY"     --arg verdict "$verdict"     --argjson requests "$requests"     --argjson native "$native"     '{schema:"ds41-transfer-native-l2-m1-finalizer-v1",status:$status,terminal:$verdict,requests_used:$requests,native:$native}'     | atomic_json
  cat "$OUT"
  exit 0
fi

[[ "$verdict" == L2_M1_FAIL ]] || {
  echo "unknown L2 M1 terminal status: $verdict" >&2
  exit 4
}

nstate=$(jq -r .state <<<"$native")
if [[ "$nstate" == READY || "$nstate" == STARTING || "$nstate" == ERROR ]]; then
  bash "$CTL" off >/dev/null
elif [[ "$nstate" != OFF ]]; then
  echo "refuse failover from non-owned native state: $nstate" >&2
  exit 5
fi

bash "$DS4CTL" on >/dev/null
deadline=$((SECONDS+1200))
dstate=''
while (( SECONDS < deadline )); do
  dstate=$(bash "$DS4CTL" status | jq -r .state)
  [[ "$dstate" == READY ]] && break
  [[ "$dstate" == ERROR ]] && break
  sleep 5
done
[[ "$dstate" == READY ]] || {
  echo "DS4 fallback did not become READY: $dstate" >&2
  exit 6
}
ds4=$(bash "$DS4CTL" status)
jq -n   --arg status "L2_M1_FAIL_DS4_READY"   --arg verdict "$verdict"   --argjson requests "$requests"   --argjson ds4 "$ds4"   '{schema:"ds41-transfer-native-l2-m1-finalizer-v1",status:$status,terminal:$verdict,requests_used:$requests,ds4:$ds4}'   | atomic_json
cat "$OUT"
