#!/usr/bin/env bash
set -Eeuo pipefail

REL=${DS41_ATTN002_REL:-/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-attnparity002-m1}
DS4CTL=/home/funboy/StrixHaloClusterDS41/scripts/ds4-document-controller.sh
PAIR="$REL/runtime/ds41/pair.sh"
OWNER=/home/funboy/.local/state/strix-cluster/owner.json
CFG=${DS41_ATTN002_CFG:-$REL/config.attention-parity-002-m1.json}
ATTEMPT=${DS41_ATTN002_ATTEMPT:-ds41-v41-attention-parity-002-m1}
MODEL=${DS41_ATTN002_MODEL:-DeepSeek-V4.1-Flash-Q2-AttnParity002-M1}
PROFILE_STATUS=${DS41_ATTN002_PROFILE_STATUS:-DS41_V41_ATTENTION_PARITY_002_M1}
RANK_PORT=18220
PAIR_URL=http://127.0.0.1:18221
PAIR_UNIT=${DS41_ATTN002_PAIR_UNIT:-ds41-attnparity002-pair.service}
GENERIC_PAIR=ds41-haloclu-pair.service
RAW=${DS41_ATTN002_RAW:-/home/funboy/reports/DS41-ATTENTION-PARITY-002}
PEER=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)
mkdir -p "$RAW"

health_code() {
  local c
  c=$(curl --noproxy '*' -sS -o /dev/null --max-time 3 -w '%{http_code}' "$1/health" 2>/dev/null || true)
  printf '%s' "${c:-000}"
}
models_json() {
  curl --noproxy '*' -sS --max-time 3 "$PAIR_URL/v1/models" 2>/dev/null || printf '{}'
}
model_ok() {
  models_json | jq -e --arg m "$MODEL" '.data | type=="array" and any(.[]; .id==$m)' >/dev/null 2>&1
}
unit_active_local() { systemctl --user is-active "$1" 2>/dev/null || true; }
unit_active_peer() { "${PEER[@]}" "systemctl --user is-active '$1' 2>/dev/null || true" 2>/dev/null || true; }

status() {
  local owner state epoch a0 a1 pair generic h0 h1 hp ds4 mids
  owner=$(jq -r '.owner // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  state=$(jq -r '.state // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  epoch=$(jq -r '.epoch // ""' "$OWNER" 2>/dev/null || true)
  a0=$(unit_active_local ds41-rank0.service)
  a1=$(unit_active_peer ds41-rank1.service)
  pair=$(unit_active_local "$PAIR_UNIT")
  generic=$(unit_active_local "$GENERIC_PAIR")
  h0=$(health_code "http://10.55.0.1:$RANK_PORT")
  h1=$(health_code "http://10.55.0.2:$RANK_PORT")
  hp=$(health_code "$PAIR_URL")
  mids=$(models_json)
  ds4=$("$DS4CTL" status)
  jq -n     --arg owner "$owner" --arg owner_state "$state" --arg epoch "$epoch"     --arg rank0 "$a0" --arg rank1 "$a1" --arg pair "$pair" --arg generic "$generic"     --arg h0 "$h0" --arg h1 "$h1" --arg hp "$hp" --arg model "$MODEL"     --argjson models "$mids" --argjson ds4 "$ds4"     '{campaign:"DS41_V41_ATTENTION_PARITY_002",release:"'"$REL"'",attempt:"'"$ATTEMPT"'",
      model:$model,owner:$owner,owner_state:$owner_state,epoch:$epoch,
      rank0:{active:$rank0,http:$h0},rank1:{active:$rank1,http:$h1},
      pair:{active:$pair,http:$hp,models:$models},generic_pair:$generic,ds4:$ds4}'
}

wait_ds4_ready() {
  local deadline=$((SECONDS+1300)) st
  while (( SECONDS < deadline )); do
    st=$("$DS4CTL" status)
    case $(jq -r .state <<<"$st") in
      READY) printf '%s\n' "$st"; return 0 ;;
      ERROR) printf '%s\n' "$st" >&2; return 1 ;;
    esac
    sleep 4
  done
  echo "DS4 restore timeout" >&2
  return 1
}

stop_candidate_pair_if_present() {
  systemctl --user stop "$PAIR_UNIT" 2>/dev/null || true
}

restore_generic_pair() {
  systemctl --user reset-failed "$GENERIC_PAIR" 2>/dev/null || true
  systemctl --user start "$GENERIC_PAIR" 2>/dev/null || true
}

fallback_after_failure() {
  local rc=$?
  trap - ERR
  echo "ATTN_PARITY_002_STARTUP_FAILURE rc=$rc" >&2
  stop_candidate_pair_if_present
  local o s
  o=$(jq -r '.owner // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  s=$(jq -r '.state // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  if [[ "$o" == DS41 && ( "$s" == RUNNING || "$s" == STARTING ) ]]; then
    DS41_ROOT="$REL" bash "$PAIR" stop || true
  fi
  o=$(jq -r '.owner // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  s=$(jq -r '.state // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  restore_generic_pair
  if [[ "$o" == NONE && "$s" == OFF ]]; then
    "$DS4CTL" on >/dev/null || true
    wait_ds4_ready >/dev/null || true
  fi
  status >"$RAW/startup-failure-status.json" || true
  exit "$rc"
}

on() {
  [[ -f "$REL/runtime/ds41/attention-parity-002-release.txt" ]]
  [[ -f "$REL/runtime/ds41/transfer-ds4-native-001-l2-release.json" ]]
  [[ -f "$REL/runtime/cluster.ds41-k2.json" ]]
  [[ -f "$CFG" ]]
  jq -e --arg m "$MODEL" --arg p "$PROFILE_STATUS" '.model==$m and .profile_status==$p' "$CFG" >/dev/null

  local dstate owner state a0 a1 epoch deadline
  dstate=$("$DS4CTL" status | jq -r .state)
  if [[ "$dstate" == READY ]]; then
    "$DS4CTL" off >/dev/null
  elif [[ "$dstate" != OFF ]]; then
    echo "refuse candidate start from DS4 state $dstate" >&2
    exit 5
  fi

  owner=$(jq -r '.owner // "UNKNOWN"' "$OWNER")
  state=$(jq -r '.state // "UNKNOWN"' "$OWNER")
  a0=$(unit_active_local ds41-rank0.service)
  a1=$(unit_active_peer ds41-rank1.service)
  [[ "$owner" == NONE && "$state" == OFF && "$a0" != active && "$a1" != active ]] || {
    echo "native not cleanly OFF owner=$owner state=$state rank=$a0/$a1" >&2
    exit 6
  }

  systemctl --user stop "$GENERIC_PAIR" 2>/dev/null || true
  [[ $(unit_active_local "$GENERIC_PAIR") != active ]] || {
    echo "generic pair did not stop" >&2; exit 7;
  }

  trap fallback_after_failure ERR

  epoch=$(date +%s%N)
  DS41_ROOT="$REL"   DS41_RUN_MODE=api   DS41_ATTEMPT_NAME="$ATTEMPT"   DS41_RUNTIME_MAX_SEC=infinity   DS41_ENGRAM_RANDOM_ADVICE=1   DS41_ENGRAM_READ_WORKERS=4   DS41_ENGRAM_PARALLEL_MIN_ROWS=256   DS41_PREFILL_TELEMETRY=1   DS41_API_MAX_MODEL_LEN=16384   DS41_CANONICAL_PREFILL_TOPK=1   DS41_DS4_MMQ_PREFILL=1   DS41_DS4_MMQ_MIN_TOKENS=128   DS41_DS4_MMQ_MAX_TOKENS=1024   DS41_MOE_PREFILL_BLOCK_M=4   bash "$PAIR" start "$epoch" "$RANK_PORT" | tee "$RAW/pair-start.txt"

  deadline=$((SECONDS+650))
  while (( SECONDS < deadline )); do
    h0=$(health_code "http://10.55.0.1:$RANK_PORT")
    h1=$(health_code "http://10.55.0.2:$RANK_PORT")
    a0=$(unit_active_local ds41-rank0.service)
    a1=$(unit_active_peer ds41-rank1.service)
    if [[ "$h0" == 200 && "$h1" == 200 ]]; then break; fi
    if [[ "$a0" != active || "$a1" != active ]]; then
      echo "rank startup terminalized active=$a0/$a1 health=$h0/$h1" >&2
      false
    fi
    sleep 4
  done
  [[ "$(health_code "http://10.55.0.1:$RANK_PORT")" == 200 &&
     "$(health_code "http://10.55.0.2:$RANK_PORT")" == 200 ]] || {
    echo "rank readiness timeout" >&2; false;
  }

  : >"$RAW/pair-coordinator.log"
  systemctl --user reset-failed "$PAIR_UNIT" 2>/dev/null || true
  systemd-run --user --unit="${PAIR_UNIT%.service}" --collect     --property=Restart=no --property=KillMode=control-group --property=RuntimeMaxSec=infinity     --property="WorkingDirectory=$REL"     --property="StandardOutput=append:$RAW/pair-coordinator.log"     --property="StandardError=append:$RAW/pair-coordinator.log"     "$REL/bin/strixglm" cluster --config "$CFG" serve-pair

  deadline=$((SECONDS+90))
  while (( SECONDS < deadline )); do
    if [[ "$(health_code "$PAIR_URL")" == 200 ]] && model_ok; then break; fi
    [[ $(unit_active_local "$PAIR_UNIT") == active ]] || {
      echo "paired coordinator terminalized" >&2; false;
    }
    sleep 2
  done
  [[ "$(health_code "$PAIR_URL")" == 200 ]] && model_ok || {
    echo "paired coordinator identity/readiness timeout" >&2; false;
  }

  trap - ERR
  status | tee "$RAW/ready.json"
}

off() {
  local owner state deadline busy
  owner=$(jq -r '.owner // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  state=$(jq -r '.state // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  if [[ "$owner" == NONE && "$state" == OFF ]]; then
    stop_candidate_pair_if_present
    restore_generic_pair
    status
    return 0
  fi
  [[ "$owner" == DS41 ]] || { echo "refuse stop foreign owner=$owner state=$state" >&2; exit 6; }

  deadline=$((SECONDS+650)); busy=true
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' -fsS --max-time 3 "$PAIR_URL/health" >/tmp/attn002-pair-health.json 2>/dev/null; then
      busy=$(jq -r '.busy // false' /tmp/attn002-pair-health.json)
    else
      busy=false
    fi
    [[ "$busy" == false ]] && break
    sleep 2
  done
  [[ "$busy" == false ]] || { echo "candidate pair did not drain" >&2; exit 7; }
  stop_candidate_pair_if_present
  DS41_ROOT="$REL" bash "$PAIR" stop
  restore_generic_pair
  status
}

fallback_ds4() {
  local o s
  o=$(jq -r '.owner // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  s=$(jq -r '.state // "UNKNOWN"' "$OWNER" 2>/dev/null || echo UNKNOWN)
  if [[ "$o" != NONE || "$s" != OFF ]]; then off >/dev/null; else restore_generic_pair; fi
  "$DS4CTL" on >/dev/null
  wait_ds4_ready
}

case ${1:-} in
  status) status ;;
  on) on ;;
  off) off ;;
  fallback-ds4) fallback_ds4 ;;
  *) echo "usage: attention-parity-002-controller.sh status|on|off|fallback-ds4" >&2; exit 2 ;;
esac
