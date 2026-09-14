#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd -P)
SHARED=/home/funboy/.local/state/strix-cluster
OWNER=$SHARED/owner.json
STATE=/home/funboy/.local/state/haloclu-ds41
PAIR_STATE=$STATE/cluster
PRESET=dspark-k2-gfx1151
MODEL=DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2
RANK_PORT=18220
PAIR_URL=http://127.0.0.1:18221
RELEASE_ID=$(cat "$ROOT/.source-commit" 2>/dev/null || git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
ssh_base=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)
mkdir -p "$STATE" "$PAIR_STATE"

unit_show() {
  local rank=$1 unit=ds41-rank$1.service
  if [[ $rank == 0 ]]; then systemctl --user show "$unit" -p LoadState -p ActiveState -p SubState -p MainPID -p InvocationID -p Environment 2>/dev/null || return 1
  else "${ssh_base[@]}" systemctl --user show "$unit" -p LoadState -p ActiveState -p SubState -p MainPID -p InvocationID -p Environment 2>/dev/null || return 1
  fi
}
field() { sed -n "s/^$2=//p" <<<"$1" | tail -1; }
active() { local a p; a=$(field "$1" ActiveState); p=$(field "$1" MainPID); [[ "$a" == active || "$a" == activating || "$a" == deactivating || ( -n "$p" && "$p" != 0 ) ]]; }
serving_env() { local e; e=$(field "$1" Environment); [[ " $e " == *" DS41_SERVING_PRESET=$PRESET "* || " $e " == *" \"DS41_SERVING_PRESET=$PRESET\" "* ]] && [[ " $e " == *" DS41_REAL_DSPARK_K=2 "* || " $e " == *" \"DS41_REAL_DSPARK_K=2\" "* ]]; }
health_code() { local c; c=$(curl --noproxy '*' -sS -o /dev/null --max-time 3 -w '%{http_code}' "$1/health" 2>/dev/null || true); printf '%s' "${c:-000}"; }
status_json() {
  local u0 u1 peer_unknown=0
  u0=$(unit_show 0 || true); u1=$(unit_show 1 || true)
  [[ -n "$u1" ]] || peer_unknown=1
  local owner=NONE ostate=OFF epoch='' detail=''
  if [[ -r "$OWNER" ]]; then owner=$(jq -r '.owner // ""' "$OWNER" 2>/dev/null || echo UNKNOWN); ostate=$(jq -r '.state // ""' "$OWNER" 2>/dev/null || echo UNKNOWN); epoch=$(jq -r '.epoch // ""' "$OWNER" 2>/dev/null || true); detail=$(jq -r '.detail // ""' "$OWNER" 2>/dev/null || true); fi
  local a0=false a1=false s0=false s1=false
  active "$u0" && a0=true || true; active "$u1" && a1=true || true
  serving_env "$u0" && s0=true || true; serving_env "$u1" && s1=true || true
  local state=ERROR reason=''
  local h0=000 h1=000 hp=000
  if (( peer_unknown )); then state=ERROR; reason='NODE02 state UNKNOWN'
  elif [[ "$owner" == NONE && "$ostate" == OFF && "$a0" == false && "$a1" == false ]]; then state=OFF
  elif [[ "$owner" == DS41 && ( "$ostate" == RUNNING || "$ostate" == STARTING ) ]]; then
    if [[ "$s0" == true || "$s1" == true ]]; then
      if [[ "$a0" == true && "$a1" == true && "$s0" == true && "$s1" == true ]]; then
        h0=$(health_code http://10.55.0.1:$RANK_PORT); h1=$(health_code http://10.55.0.2:$RANK_PORT); hp=$(health_code "$PAIR_URL")
        if [[ "$h0" == 200 && "$h1" == 200 && "$hp" == 200 ]]; then state=READY; else state=STARTING; reason="rank/backend health $h0/$h1/$hp"; fi
      else state=STARTING; reason='serving ownership exists but both serving ranks are not active yet'; fi
    else state=RESEARCH_BUSY; reason="pair owned by research: ${detail:-$ostate}"; fi
  elif [[ "$owner" == DS41 && "$ostate" == UNRECONCILED ]]; then state=ERROR; reason="pair unreconciled: $detail"
  else state=ERROR; reason="unexpected owner/unit state owner=$owner state=$ostate rank0=$a0 rank1=$a1"
  fi
  jq -n --arg state "$state" --arg preset "$PRESET" --arg release "$RELEASE_ID" --arg owner "$owner" --arg owner_state "$ostate" --arg epoch "$epoch" --arg reason "$reason" --arg h0 "$h0" --arg h1 "$h1" --arg hp "$hp" --argjson r0 "$a0" --argjson r1 "$a1" '{state:$state,preset:$preset,release_id:$release,owner:$owner,owner_state:$owner_state,epoch:$epoch,reason:$reason,ranks:[{rank:0,active:$r0,health_http:$h0},{rank:1,active:$r1,health_http:$h1}],paired_backend_http:$hp}'
}
state_only() { status_json | jq -r .state; }
wait_ready() {
  local deadline=$((SECONDS+1200)) state
  while (( SECONDS < deadline )); do state=$(state_only); [[ "$state" == READY ]] && return 0; [[ "$state" == ERROR || "$state" == RESEARCH_BUSY ]] && { status_json >&2; return 1; }; sleep 5; done
  echo 'serving readiness timeout' >&2; status_json >&2; return 1
}
smoke() {
  local tmp="$STATE/last-on-smoke.json" body
  body=$(cat <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"Compute 17*19. Return only the integer."}],"temperature":0,"max_tokens":128,"seed":1,"stream":false,"chat_template_kwargs":{"reasoning_effort":"none"}}
JSON
)
  curl --noproxy '*' -fsS --max-time 180 -H 'Content-Type: application/json' -d "$body" "$PAIR_URL/v1/chat/completions" >"$tmp"
  jq -e '.choices | (length == 1 and .[0].message.content == "323")' "$tmp" >/dev/null
}
start_serving() {
  local st; st=$(state_only)
  [[ "$st" == READY ]] && { status_json; return 0; }
  [[ "$st" == OFF ]] || { echo "cannot start K2 serving from state $st" >&2; status_json >&2; return 4; }
  systemctl --user is-active --quiet ds41-haloclu-pair.service || { echo 'ds41-haloclu-pair.service must be active before model ON' >&2; return 5; }
  local epoch; epoch=$(date +%s%N)
  DS41_ROOT="$ROOT" DS41_RUN_MODE=api DS41_ATTEMPT_NAME=serving-k2 DS41_RUNTIME_MAX_SEC=infinity \
  DS41_SERVING_PRESET="$PRESET" DS41_SERVING_RELEASE_ID="$RELEASE_ID" DS41_REAL_DSPARK_K=2 \
  DS41_ENGRAM_RANDOM_ADVICE=1 bash "$ROOT/runtime/ds41/pair.sh" start "$epoch" "$RANK_PORT"
  wait_ready
  smoke
  status_json
}
stop_serving() {
  local st; st=$(state_only)
  [[ "$st" == OFF ]] && { status_json; return 0; }
  [[ "$st" != RESEARCH_BUSY ]] || { echo 'refusing to stop pair owned by research' >&2; status_json >&2; return 4; }
  [[ "$st" == READY || "$st" == STARTING || "$st" == ERROR ]] || { echo "cannot stop state $st" >&2; return 4; }
  # Drain policy: no new client is admitted by gateway action state; paired backend must become idle.
  local deadline=$((SECONDS+650)) busy=true
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' -fsS --max-time 3 "$PAIR_URL/health" >"$STATE/.pair-health.json" 2>/dev/null; then busy=$(jq -r '.busy // false' "$STATE/.pair-health.json"); else busy=false; fi
    [[ "$busy" == false ]] && break
    sleep 2
  done
  [[ "$busy" == false ]] || { echo 'paired request did not drain before OFF deadline' >&2; return 6; }
  DS41_ROOT="$ROOT" bash "$ROOT/runtime/ds41/pair.sh" stop
  local final; final=$(state_only); [[ "$final" == OFF ]] || { status_json >&2; return 7; }
  # If a previous abnormal frontend left a poison marker, whole-pair OFF makes it safe to clear and restart only the coordinator.
  if [[ -e "$PAIR_STATE/pair-poison.json" ]]; then rm -f "$PAIR_STATE/pair-poison.json"; systemctl --user restart ds41-haloclu-pair.service; fi
  status_json
}
case ${1:-} in
  status) status_json ;;
  on) start_serving ;;
  off) stop_serving ;;
  *) echo 'usage: serve-controller.sh status|on|off' >&2; exit 2 ;;
esac
