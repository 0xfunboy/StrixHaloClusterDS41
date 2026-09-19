#!/usr/bin/env bash
set -Eeuo pipefail

REL=/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001-anon1
DS4CTL=/home/funboy/StrixHaloClusterDS41/scripts/ds4-document-controller.sh
PAIR="$REL/runtime/ds41/pair.sh"
OWNER=/home/funboy/.local/state/strix-cluster/owner.json
ATTEMPT=transfer-ds4-native-001-l2-m1-anon1
RANK_PORT=18220
PAIR_URL=http://127.0.0.1:18221
PEER=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)

health_code() {
  local c
  c=$(curl --noproxy '*' -sS -o /dev/null --max-time 3 -w '%{http_code}' "$1/health" 2>/dev/null || true)
  printf '%s' "${c:-000}"
}

unit_env_local() {
  systemctl --user show ds41-rank0.service -p ActiveState -p MainPID -p InvocationID -p Environment 2>/dev/null || true
}
unit_env_peer() {
  "${PEER[@]}" systemctl --user show ds41-rank1.service -p ActiveState -p MainPID -p InvocationID -p Environment 2>/dev/null || true
}
field() { sed -n "s/^$2=//p" <<<"$1" | tail -1; }

native_status() {
  local u0 u1 a0 a1 e0 e1 owner state epoch h0=000 h1=000 hp=000 result reason=''
  u0=$(unit_env_local); u1=$(unit_env_peer)
  a0=$(field "$u0" ActiveState); a1=$(field "$u1" ActiveState)
  e0=$(field "$u0" Environment); e1=$(field "$u1" Environment)
  owner=$(jq -r '.owner // "NONE"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  state=$(jq -r '.state // "OFF"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  epoch=$(jq -r '.epoch // ""' "$OWNER" 2>/dev/null || true)

  if [[ "$owner" == NONE && "$state" == OFF && "$a0" != active && "$a1" != active ]]; then
    result=OFF
  elif [[ "$owner" == DS41 && ( "$state" == RUNNING || "$state" == STARTING ) ]]; then
    if [[ " $e0 " != *" DS41_ROOT=$REL "* || " $e1 " != *" DS41_ROOT=$REL "* ||
          " $e0 " != *" DS41_ATTEMPT_NAME=$ATTEMPT "* || " $e1 " != *" DS41_ATTEMPT_NAME=$ATTEMPT "* ]]; then
      result=FOREIGN
      reason='DS41 owner belongs to another release/attempt'
    elif [[ "$a0" != active || "$a1" != active ]]; then
      result=ERROR
      reason="L2 owner exists but rank active states are $a0/$a1"
    else
      h0=$(health_code "http://10.55.0.1:$RANK_PORT")
      h1=$(health_code "http://10.55.0.2:$RANK_PORT")
      hp=$(health_code "$PAIR_URL")
      if [[ "$h0" == 200 && "$h1" == 200 && "$hp" == 200 ]]; then
        result=READY
      else
        result=STARTING
        reason="health $h0/$h1/$hp"
      fi
    fi
  else
    result=ERROR
    reason="owner=$owner state=$state rank=$a0/$a1"
  fi

  jq -n     --arg state "$result" --arg release "$REL" --arg attempt "$ATTEMPT"     --arg owner "$owner" --arg owner_state "$state" --arg epoch "$epoch"     --arg rank0 "$a0" --arg rank1 "$a1" --arg h0 "$h0" --arg h1 "$h1" --arg hp "$hp"     --arg reason "$reason"     '{state:$state,release:$release,attempt:$attempt,target:"Antirez-Q2",engram:"native-GGUF",dspark:false,
      owner:$owner,owner_state:$owner_state,epoch:$epoch,rank0:{active:$rank0,http:$h0},
      rank1:{active:$rank1,http:$h1},paired_http:$hp,reason:$reason}'
}

status() {
  local native ds4
  native=$(native_status)
  ds4=$("$DS4CTL" status)
  jq -n --argjson native "$native" --argjson ds4 "$ds4" '{native:$native,ds4:$ds4}'
}

on() {
  [[ -f "$REL/runtime/ds41/transfer-ds4-native-001-l2-release.json" ]] || {
    echo "L2 release not prepared" >&2
    exit 2
  }
  local nstate dstate
  nstate=$(native_status | jq -r .state)
  [[ "$nstate" == READY ]] && { status; return 0; }
  [[ "$nstate" == OFF ]] || { echo "L2 start refused from native state $nstate" >&2; status >&2; exit 4; }
  dstate=$("$DS4CTL" status | jq -r .state)
  if [[ "$dstate" == READY ]]; then
    "$DS4CTL" off >/dev/null
  elif [[ "$dstate" != OFF ]]; then
    echo "L2 start refused from DS4 state $dstate" >&2
    exit 5
  fi
  local epoch
  epoch=$(date +%s%N)
  DS41_ROOT="$REL"   DS41_RUN_MODE=api   DS41_ATTEMPT_NAME="$ATTEMPT"   DS41_RUNTIME_MAX_SEC=infinity   DS41_ENGRAM_RANDOM_ADVICE=1   DS41_ENGRAM_READ_WORKERS=4   DS41_ENGRAM_PARALLEL_MIN_ROWS=256   DS41_PREFILL_TELEMETRY=1   DS41_API_MAX_MODEL_LEN=16384   DS41_CANONICAL_PREFILL_TOPK=1   DS41_DS4_MMQ_PREFILL=1   DS41_DS4_MMQ_MIN_TOKENS=128   DS41_DS4_MMQ_MAX_TOKENS=1024   DS41_MOE_PREFILL_BLOCK_M=4   bash "$PAIR" start "$epoch" "$RANK_PORT"
  status
}

off() {
  local st
  st=$(native_status | jq -r .state)
  [[ "$st" == OFF ]] && { status; return 0; }
  [[ "$st" != FOREIGN ]] || { echo 'refuse to stop foreign DS41 owner' >&2; exit 4; }
  local deadline=$((SECONDS+650)) busy=true
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' -fsS --max-time 3 "$PAIR_URL/health" >/tmp/ds41-transfer-l2-health.json 2>/dev/null; then
      busy=$(jq -r '.busy // false' /tmp/ds41-transfer-l2-health.json)
    else
      busy=false
    fi
    [[ "$busy" == false ]] && break
    sleep 2
  done
  [[ "$busy" == false ]] || { echo 'L2 paired request did not drain' >&2; exit 6; }
  DS41_ROOT="$REL" bash "$PAIR" stop
  status
}

fallback_ds4() {
  local st
  st=$(native_status | jq -r .state)
  if [[ "$st" != OFF ]]; then off >/dev/null; fi
  "$DS4CTL" on
}

case ${1:-} in
  status) status ;;
  on) on ;;
  off) off ;;
  fallback-ds4) fallback_ds4 ;;
  *) echo 'usage: transfer-ds4-native-001-l2-controller.sh status|on|off|fallback-ds4' >&2; exit 2 ;;
esac
