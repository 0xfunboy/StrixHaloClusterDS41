#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${DS41_ROOT:-/home/funboy/StrixHaloClusterDS41}
attempt=${1:?attempt-name}
epoch=${2:?epoch}
max_sec=${3:-2400}
log="$ROOT/reports/DS41-Q2-001/$attempt/supervisor.log"
mkdir -p "$(dirname "$log")"
exec >>"$log" 2>&1
printf 'SUPERVISOR_START attempt=%s epoch=%s max_sec=%s at=%s\n' "$attempt" "$epoch" "$max_sec" "$(date -u +%FT%TZ)"
deadline=$((SECONDS+max_sec))
last=''
while (( SECONDS < deadline )); do
  owner=$(jq -r '.owner // ""' /home/funboy/.local/state/strix-cluster/owner.json 2>/dev/null || true)
  state=$(jq -r '.state // ""' /home/funboy/.local/state/strix-cluster/owner.json 2>/dev/null || true)
  got_epoch=$(jq -r '.epoch // ""' /home/funboy/.local/state/strix-cluster/owner.json 2>/dev/null || true)
  l=$(systemctl --user is-active ds41-rank0.service 2>/dev/null || true)
  r=$(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb 'systemctl --user is-active ds41-rank1.service 2>/dev/null || true' 2>/dev/null || echo unknown)
  cur="owner=$owner state=$state epoch=$got_epoch rank0=$l rank1=$r"
  if [[ "$cur" != "$last" ]]; then printf 'SUPERVISOR_STATE %s at=%s\n' "$cur" "$(date -u +%FT%TZ)"; last=$cur; fi
  if [[ "$owner" != DS41 || "$got_epoch" != "$epoch" ]]; then
    printf 'SUPERVISOR_EXIT receipt_changed %s\n' "$cur"
    exit 0
  fi
  if [[ "$l" != active || "$r" != active ]]; then
    printf 'SUPERVISOR_TERMINAL rank_transition %s\n' "$cur"
    set +e
    "$ROOT/runtime/ds41/pair.sh" stop
    rc=$?
    set -e
    printf 'SUPERVISOR_CLEANUP rc=%s at=%s\n' "$rc" "$(date -u +%FT%TZ)"
    exit "$rc"
  fi
  sleep 10
done
printf 'SUPERVISOR_TIMEOUT attempt=%s epoch=%s at=%s\n' "$attempt" "$epoch" "$(date -u +%FT%TZ)"
set +e
"$ROOT/runtime/ds41/pair.sh" stop
rc=$?
set -e
printf 'SUPERVISOR_TIMEOUT_CLEANUP rc=%s\n' "$rc"
exit "$rc"
