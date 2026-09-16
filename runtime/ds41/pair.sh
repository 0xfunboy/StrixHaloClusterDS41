#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${DS41_ROOT:-/home/funboy/StrixHaloClusterDS41}
SHARED_STATE=${DS41_SHARED_STATE:-/home/funboy/.local/state/strix-cluster}
OWNER_FILE="$SHARED_STATE/owner.json"
CONTROL_LOCK="$SHARED_STATE/compute.lock"
RUNTIME_MAX_SEC=${DS41_RUNTIME_MAX_SEC:-5400}
STOP_TIMEOUT_SEC=${DS41_STOP_TIMEOUT_SEC:-30}
ssh_base=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)

[[ "$(hostname)" == 01-EVO-X3 ]] || { echo 'pair control must run on 01-EVO-X3' >&2; exit 2; }
mkdir -p "$SHARED_STATE"

peer() {
  if [[ -n "${DS41_TEST_PEER_CMD:-}" ]]; then
    "$DS41_TEST_PEER_CMD" "$@"
  else
    "${ssh_base[@]}" "$@"
  fi
}
peer_identity() { [[ "$(peer hostname 2>/dev/null)" == 02-EVO-X3 ]]; }
now_utc() { date -u +%FT%TZ; }
new_nonce() { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; }

atomic_owner() {
  local owner=$1 state=$2 epoch=$3 detail=$4 r0_inv=$5 r0_nonce=$6 r1_inv=$7 r1_nonce=$8
  local tmp="$SHARED_STATE/.owner.$$.tmp"
  jq -n \
    --arg owner "$owner" --arg state "$state" --arg epoch "$epoch" --arg detail "$detail" \
    --arg r0_inv "$r0_inv" --arg r0_nonce "$r0_nonce" --arg r1_inv "$r1_inv" --arg r1_nonce "$r1_nonce" \
    --arg updated "$(now_utc)" \
    '{schema:"strix-cluster-compute-v2",owner:$owner,state:$state,epoch:$epoch,detail:$detail,updated:$updated,
      ranks:{"0":{unit:"ds41-rank0.service",invocation_id:$r0_inv,nonce:$r0_nonce},
             "1":{unit:"ds41-rank1.service",invocation_id:$r1_inv,nonce:$r1_nonce}}}' >"$tmp"
  chmod 600 "$tmp"
  mv -f "$tmp" "$OWNER_FILE"
}

owner_field() {
  local expr=$1 default=${2:-}
  [[ -r "$OWNER_FILE" ]] || { printf '%s' "$default"; return 0; }
  jq -er "$expr // empty" "$OWNER_FILE" 2>/dev/null || printf '%s' "$default"
}

probe_script='unit=$1
out=$(systemctl --user show "$unit" -p LoadState -p ActiveState -p SubState -p MainPID -p InvocationID -p ControlGroup -p Environment 2>&1) || { printf "ProbeError=%s\n" "$out"; exit 40; }
printf "%s\n" "$out"
cg=$(printf "%s\n" "$out" | sed -n "s/^ControlGroup=//p")
if [[ -n "$cg" && -r "/sys/fs/cgroup${cg}/cgroup.procs" ]]; then
  pids=$(tr "\n" "," <"/sys/fs/cgroup${cg}/cgroup.procs" | sed "s/,$//")
  printf "CgroupPIDs=%s\n" "$pids"
else
  printf "CgroupPIDs=\n"
fi'

probe_local() {
  /bin/bash -c "$probe_script" _ "$1" 2>&1
}
probe_peer() {
  peer /bin/bash -c "$(printf '%q' "$probe_script")" _ "$1" 2>&1
}

field() {
  local raw=$1 key=$2
  printf '%s\n' "$raw" | sed -n "s/^${key}=//p" | tail -1
}

classify_probe() {
  local raw=$1
  local load active pid cg cgpids
  load=$(field "$raw" LoadState); active=$(field "$raw" ActiveState); pid=$(field "$raw" MainPID)
  cg=$(field "$raw" ControlGroup); cgpids=$(field "$raw" CgroupPIDs)
  [[ -n "$load" && -n "$active" && -n "$pid" ]] || { echo UNKNOWN; return; }
  if [[ "$pid" != 0 || "$active" == active || "$active" == activating || "$active" == deactivating || -n "$cgpids" ]]; then
    echo ACTIVE; return
  fi
  if [[ ( "$active" == inactive || "$active" == failed ) && "$pid" == 0 && -z "$cgpids" ]]; then
    echo OFF_VERIFIED; return
  fi
  echo UNKNOWN
}

safe_probe_local() {
  local raw rc
  set +e; raw=$(probe_local "$1"); rc=$?; set -e
  (( rc == 0 )) || { printf 'UNKNOWN\n%s\n' "$raw"; return 0; }
  printf '%s\n%s\n' "$(classify_probe "$raw")" "$raw"
}
safe_probe_peer() {
  local raw rc
  set +e; raw=$(probe_peer "$1"); rc=$?; set -e
  (( rc == 0 )) || { printf 'UNKNOWN\n%s\n' "$raw"; return 0; }
  printf '%s\n%s\n' "$(classify_probe "$raw")" "$raw"
}

probe_state() { printf '%s\n' "$1" | head -1; }
probe_body() { printf '%s\n' "$1" | tail -n +2; }

validate_active_identity() {
  local probe=$1 epoch=$2 nonce=$3 invocation=$4
  [[ "$(probe_state "$probe")" == ACTIVE ]] || return 1
  local raw env got_inv
  raw=$(probe_body "$probe"); env=$(field "$raw" Environment); got_inv=$(field "$raw" InvocationID)
  [[ -n "$epoch" && -n "$nonce" && -n "$invocation" && "$got_inv" == "$invocation" ]] || return 1
  [[ " $env " == *" DS41_OWNER_EPOCH=$epoch "* || " $env " == *" \"DS41_OWNER_EPOCH=$epoch\" "* ]] || return 1
  [[ " $env " == *" DS41_OWNER_NONCE=$nonce "* || " $env " == *" \"DS41_OWNER_NONCE=$nonce\" "* ]] || return 1
}

acquire_control() {
  exec 9>"$CONTROL_LOCK"
  flock -n 9 || { echo 'cluster lifecycle operation already in progress' >&2; exit 75; }
}

ensure_startable_receipt() {
  if [[ ! -r "$OWNER_FILE" ]]; then
    atomic_owner NONE OFF '' 'initialized after both nodes are verified OFF' '' '' '' ''
    return 0
  fi
  local schema owner state
  schema=$(owner_field '.schema'); owner=$(owner_field '.owner'); state=$(owner_field '.state')
  [[ "$schema" == strix-cluster-compute-v2 && "$owner" == NONE && "$state" == OFF ]] || {
    echo "persistent cluster state is not reconciled OFF (schema=$schema owner=$owner state=$state); run reconcile only after both peers are verified OFF" >&2
    exit 6
  }
}

persist_unreconciled() {
  local reason=$1
  local owner epoch r0_inv r0_nonce r1_inv r1_nonce
  owner=$(owner_field '.owner' DS41); [[ "$owner" == NONE ]] && owner=DS41
  epoch=$(owner_field '.epoch'); r0_inv=$(owner_field '.ranks["0"].invocation_id'); r0_nonce=$(owner_field '.ranks["0"].nonce')
  r1_inv=$(owner_field '.ranks["1"].invocation_id'); r1_nonce=$(owner_field '.ranks["1"].nonce')
  atomic_owner "$owner" UNRECONCILED "$epoch" "$reason" "$r0_inv" "$r0_nonce" "$r1_inv" "$r1_nonce"
}

wait_active_local() {
  local unit=$1 deadline=$((SECONDS+8)) probe
  while (( SECONDS < deadline )); do
    probe=$(safe_probe_local "$unit")
    [[ "$(probe_state "$probe")" == ACTIVE ]] && { printf '%s\n' "$probe"; return 0; }
    [[ "$(probe_state "$probe")" == UNKNOWN ]] && break
    sleep .2
  done
  return 1
}
wait_active_peer() {
  local unit=$1 deadline=$((SECONDS+8)) probe
  while (( SECONDS < deadline )); do
    probe=$(safe_probe_peer "$unit")
    [[ "$(probe_state "$probe")" == ACTIVE ]] && { printf '%s\n' "$probe"; return 0; }
    [[ "$(probe_state "$probe")" == UNKNOWN ]] && break
    sleep .2
  done
  return 1
}

start_unit_local() {
  local epoch=$1 port=$2 nonce=$3 log=$4
  systemctl --user reset-failed ds41-rank0.service 2>/dev/null || true
  systemd-run --user --unit=ds41-rank0 --collect \
    --setenv="DS41_OWNER_EPOCH=$epoch" --setenv="DS41_OWNER_NONCE=$nonce" \
    --setenv="DS41_RUN_MODE=${DS41_RUN_MODE:-api}" --setenv="DS41_ATTEMPT_NAME=${DS41_ATTEMPT_NAME:-}" \
    --setenv="DS41_ROOT=$ROOT" --setenv="DS41_SERVING_PRESET=${DS41_SERVING_PRESET:-}" --setenv="DS41_SERVING_RELEASE_ID=${DS41_SERVING_RELEASE_ID:-}" \
    --setenv="DS41_REAL_DSPARK_K=${DS41_REAL_DSPARK_K:-}" --setenv="DS41_ENGRAM_RANDOM_ADVICE=${DS41_ENGRAM_RANDOM_ADVICE:-1}" \
    --setenv="DS41_PREFILL_TELEMETRY=${DS41_PREFILL_TELEMETRY:-0}" --setenv="DS41_API_MAX_MODEL_LEN=${DS41_API_MAX_MODEL_LEN:-65664}" \
    --setenv="DS41_KV_CAPTURE_DIR=${DS41_KV_CAPTURE_DIR:-}" --setenv="DS41_KV_CAPTURE_PROMPT_TOKENS=${DS41_KV_CAPTURE_PROMPT_TOKENS:-1546}" \
    --setenv="DS41_ROUTE_CAPTURE_DIR=${DS41_ROUTE_CAPTURE_DIR:-}" --setenv="DS41_ROUTE_CAPTURE_CALLS=${DS41_ROUTE_CAPTURE_CALLS:-0,20,40,60}" --setenv="DS41_ROUTE_CAPTURE_PROMPT_TOKENS=${DS41_ROUTE_CAPTURE_PROMPT_TOKENS:-1588}" --setenv="DS41_MOE_PREFILL_BLOCK_M=${DS41_MOE_PREFILL_BLOCK_M:-4}" \
    --setenv="DS41_REPEAT_DIAG_DIR=${DS41_REPEAT_DIAG_DIR:-}" --setenv="DS41_REPEAT_DIAG_PROMPT_TOKENS=${DS41_REPEAT_DIAG_PROMPT_TOKENS:-1588}" --setenv="DS41_REPEAT_DIAG_MAX_REQUESTS=${DS41_REPEAT_DIAG_MAX_REQUESTS:-2}" --setenv="DS41_REPEAT_DIAG_MODE=${DS41_REPEAT_DIAG_MODE:-full}" --setenv="DS41_CANONICAL_PREFILL_TOPK=${DS41_CANONICAL_PREFILL_TOPK:-0}" \
    --setenv="DS41_PREFILL_PROFILE_ONCE=${DS41_PREFILL_PROFILE_ONCE:-0}" --setenv="DS41_PREFILL_PROFILE_MIN_TOKENS=${DS41_PREFILL_PROFILE_MIN_TOKENS:-1500}" --setenv="DS41_PREFILL_PROFILE_DIR=${DS41_PREFILL_PROFILE_DIR:-}" \
    --property="RuntimeMaxSec=$RUNTIME_MAX_SEC" --property="TimeoutStopSec=$STOP_TIMEOUT_SEC" --property=KillMode=control-group --property=Restart=no \
    --property="StandardOutput=append:$log" --property="StandardError=append:$log" \
    bash "$ROOT/runtime/ds41/launch-node.sh" 0 10.55.0.1 "$epoch" "$port"
}
start_unit_peer() {
  local epoch=$1 port=$2 nonce=$3 log=$4
  peer systemctl --user reset-failed ds41-rank1.service 2>/dev/null || true
  peer systemd-run --user --unit=ds41-rank1 --collect \
    --setenv="DS41_OWNER_EPOCH=$epoch" --setenv="DS41_OWNER_NONCE=$nonce" \
    --setenv="DS41_RUN_MODE=${DS41_RUN_MODE:-api}" --setenv="DS41_ATTEMPT_NAME=${DS41_ATTEMPT_NAME:-}" \
    --setenv="DS41_ROOT=$ROOT" --setenv="DS41_SERVING_PRESET=${DS41_SERVING_PRESET:-}" --setenv="DS41_SERVING_RELEASE_ID=${DS41_SERVING_RELEASE_ID:-}" \
    --setenv="DS41_REAL_DSPARK_K=${DS41_REAL_DSPARK_K:-}" --setenv="DS41_ENGRAM_RANDOM_ADVICE=${DS41_ENGRAM_RANDOM_ADVICE:-1}" \
    --setenv="DS41_PREFILL_TELEMETRY=${DS41_PREFILL_TELEMETRY:-0}" --setenv="DS41_API_MAX_MODEL_LEN=${DS41_API_MAX_MODEL_LEN:-65664}" \
    --setenv="DS41_KV_CAPTURE_DIR=${DS41_KV_CAPTURE_DIR:-}" --setenv="DS41_KV_CAPTURE_PROMPT_TOKENS=${DS41_KV_CAPTURE_PROMPT_TOKENS:-1546}" \
    --setenv="DS41_ROUTE_CAPTURE_DIR=${DS41_ROUTE_CAPTURE_DIR:-}" --setenv="DS41_ROUTE_CAPTURE_CALLS=${DS41_ROUTE_CAPTURE_CALLS:-0,20,40,60}" --setenv="DS41_ROUTE_CAPTURE_PROMPT_TOKENS=${DS41_ROUTE_CAPTURE_PROMPT_TOKENS:-1588}" --setenv="DS41_MOE_PREFILL_BLOCK_M=${DS41_MOE_PREFILL_BLOCK_M:-4}" \
    --setenv="DS41_REPEAT_DIAG_DIR=${DS41_REPEAT_DIAG_DIR:-}" --setenv="DS41_REPEAT_DIAG_PROMPT_TOKENS=${DS41_REPEAT_DIAG_PROMPT_TOKENS:-1588}" --setenv="DS41_REPEAT_DIAG_MAX_REQUESTS=${DS41_REPEAT_DIAG_MAX_REQUESTS:-2}" --setenv="DS41_REPEAT_DIAG_MODE=${DS41_REPEAT_DIAG_MODE:-full}" --setenv="DS41_CANONICAL_PREFILL_TOPK=${DS41_CANONICAL_PREFILL_TOPK:-0}" \
    --setenv="DS41_PREFILL_PROFILE_ONCE=${DS41_PREFILL_PROFILE_ONCE:-0}" --setenv="DS41_PREFILL_PROFILE_MIN_TOKENS=${DS41_PREFILL_PROFILE_MIN_TOKENS:-1500}" --setenv="DS41_PREFILL_PROFILE_DIR=${DS41_PREFILL_PROFILE_DIR:-}" \
    --property="RuntimeMaxSec=$RUNTIME_MAX_SEC" --property="TimeoutStopSec=$STOP_TIMEOUT_SEC" --property=KillMode=control-group --property=Restart=no \
    --property="StandardOutput=append:$log" --property="StandardError=append:$log" \
    bash "$ROOT/runtime/ds41/launch-node.sh" 1 10.55.0.2 "$epoch" "$port"
}

cmd_start() {
  local epoch=${1:?epoch} port=${2:-18210} run_mode=${DS41_RUN_MODE:-api}
  [[ "$epoch" =~ ^[0-9]{10,20}$ && "$port" =~ ^[0-9]{4,5}$ ]] || { echo 'invalid epoch/port' >&2; exit 2; }
  [[ "$run_mode" == api || "$run_mode" == offline ]] || { echo 'invalid DS41_RUN_MODE' >&2; exit 2; }
  [[ "${DS41_ENGRAM_RANDOM_ADVICE:-1}" == 0 || "${DS41_ENGRAM_RANDOM_ADVICE:-1}" == 1 ]] || { echo 'invalid DS41_ENGRAM_RANDOM_ADVICE' >&2; exit 2; }
  acquire_control
  peer_identity || { persist_unreconciled 'NODE02 identity unavailable before start'; echo 'NODE02 UNKNOWN; refusing start' >&2; exit 3; }

  local schema owner state current_epoch lp rp ls rs
  schema=$(owner_field '.schema'); owner=$(owner_field '.owner'); state=$(owner_field '.state'); current_epoch=$(owner_field '.epoch')
  lp=$(safe_probe_local ds41-rank0.service); rp=$(safe_probe_peer ds41-rank1.service)
  ls=$(probe_state "$lp"); rs=$(probe_state "$rp")

  # Duplicate start is idempotent only when the persisted epoch/nonces and both
  # active InvocationIDs still describe the exact pair already running.
  if [[ "$schema" == strix-cluster-compute-v2 && "$owner" == DS41 && "$state" == RUNNING ]]; then
    local old_inv0 old_nonce0 old_inv1 old_nonce1
    old_inv0=$(owner_field '.ranks["0"].invocation_id'); old_nonce0=$(owner_field '.ranks["0"].nonce')
    old_inv1=$(owner_field '.ranks["1"].invocation_id'); old_nonce1=$(owner_field '.ranks["1"].nonce')
    if validate_active_identity "$lp" "$current_epoch" "$old_nonce0" "$old_inv0" && validate_active_identity "$rp" "$current_epoch" "$old_nonce1" "$old_inv1"; then
      echo "DS41_ALREADY_RUNNING epoch=$current_epoch"
      return 0
    fi
    persist_unreconciled "running receipt no longer matches both active units: rank0=$ls rank1=$rs"
    echo 'existing DS41 receipt is stale/unverifiable; reconcile before a new start' >&2
    exit 6
  fi

  [[ "$ls" == OFF_VERIFIED && "$rs" == OFF_VERIFIED ]] || {
    persist_unreconciled "start refused: rank0=$ls rank1=$rs"
    echo "start refused: rank0=$ls rank1=$rs" >&2; exit 4
  }
  ensure_startable_receipt

  local nonce0 nonce1 logdir log0 log1 p0 p1 inv0 inv1 attempt_name
  nonce0=$(new_nonce); nonce1=$(new_nonce)
  attempt_name=${DS41_ATTEMPT_NAME:-epoch-$epoch}
  [[ "$attempt_name" =~ ^[a-zA-Z0-9._-]{1,80}$ ]] || { echo 'invalid DS41_ATTEMPT_NAME' >&2; exit 2; }
  logdir="$ROOT/reports/DS41-Q2-001/$attempt_name"; mkdir -p "$logdir"
  peer mkdir -p "$logdir" || {
    persist_unreconciled 'NODE02 attempt log directory could not be prepared'
    echo 'NODE02 log directory unavailable; refusing start' >&2
    exit 5
  }
  log0="$logdir/rank0.log"; log1="$logdir/rank1.log"
  atomic_owner DS41 STARTING "$epoch" 'rank0 launch reserved; peer not started yet' '' "$nonce0" '' "$nonce1"

  start_unit_local "$epoch" "$port" "$nonce0" "$log0"
  p0=$(wait_active_local ds41-rank0.service) || {
    persist_unreconciled 'rank0 launch not proven ACTIVE'; echo 'rank0 launch unverified' >&2; exit 5
  }
  inv0=$(field "$(probe_body "$p0")" InvocationID)
  local env0; env0=$(field "$(probe_body "$p0")" Environment)
  [[ -n "$inv0" && " $env0 " == *"DS41_OWNER_EPOCH=$epoch"* && " $env0 " == *"DS41_OWNER_NONCE=$nonce0"* ]] || {
    persist_unreconciled 'rank0 identity receipt mismatch'; echo 'rank0 identity mismatch' >&2; exit 5
  }
  atomic_owner DS41 STARTING "$epoch" 'rank0 identity confirmed; launching peer' "$inv0" "$nonce0" '' "$nonce1"

  start_unit_peer "$epoch" "$port" "$nonce1" "$log1"
  p1=$(wait_active_peer ds41-rank1.service) || {
    persist_unreconciled 'rank1 launch not proven ACTIVE'; echo 'rank1 launch unverified' >&2; exit 5
  }
  inv1=$(field "$(probe_body "$p1")" InvocationID)
  local env1; env1=$(field "$(probe_body "$p1")" Environment)
  [[ -n "$inv1" && " $env1 " == *"DS41_OWNER_EPOCH=$epoch"* && " $env1 " == *"DS41_OWNER_NONCE=$nonce1"* ]] || {
    persist_unreconciled 'rank1 identity receipt mismatch'; echo 'rank1 identity mismatch' >&2; exit 5
  }
  atomic_owner DS41 RUNNING "$epoch" "both ranks identity-confirmed; RuntimeMaxSec=${RUNTIME_MAX_SEC}" "$inv0" "$nonce0" "$inv1" "$nonce1"
  echo "DS41_START_CONFIRMED epoch=$epoch rank0_invocation=$inv0 rank1_invocation=$inv1 runtime_max_sec=$RUNTIME_MAX_SEC"
}

cmd_stop() {
  acquire_control
  local schema owner state epoch inv0 nonce0 inv1 nonce1
  schema=$(owner_field '.schema'); owner=$(owner_field '.owner'); state=$(owner_field '.state'); epoch=$(owner_field '.epoch')
  inv0=$(owner_field '.ranks["0"].invocation_id'); nonce0=$(owner_field '.ranks["0"].nonce')
  inv1=$(owner_field '.ranks["1"].invocation_id'); nonce1=$(owner_field '.ranks["1"].nonce')
  [[ "$schema" == strix-cluster-compute-v2 && "$owner" == DS41 ]] || { echo 'DS41 v2 ownership receipt required for automated stop' >&2; exit 6; }

  local rp rs lp ls
  rp=$(safe_probe_peer ds41-rank1.service); rs=$(probe_state "$rp")
  if [[ "$rs" == UNKNOWN ]]; then persist_unreconciled 'stop refused: rank1 UNKNOWN'; echo 'rank1 UNKNOWN; rank0 left untouched' >&2; exit 3; fi
  if [[ "$rs" == ACTIVE ]] && ! validate_active_identity "$rp" "$epoch" "$nonce1" "$inv1"; then
    persist_unreconciled 'stop refused: rank1 active identity mismatch'; echo 'rank1 foreign/stale identity; rank0 left untouched' >&2; exit 4
  fi
  if [[ "$rs" == ACTIVE ]]; then
    peer systemctl --user stop ds41-rank1.service
    rp=$(safe_probe_peer ds41-rank1.service); rs=$(probe_state "$rp")
  fi
  if [[ "$rs" != OFF_VERIFIED ]]; then persist_unreconciled "rank1 stop not verified: $rs"; echo 'rank1 not proven OFF; rank0 left untouched' >&2; exit 4; fi

  lp=$(safe_probe_local ds41-rank0.service); ls=$(probe_state "$lp")
  if [[ "$ls" == UNKNOWN ]]; then persist_unreconciled 'rank0 UNKNOWN after peer OFF'; echo 'rank0 UNKNOWN' >&2; exit 4; fi
  if [[ "$ls" == ACTIVE ]] && ! validate_active_identity "$lp" "$epoch" "$nonce0" "$inv0"; then
    persist_unreconciled 'rank0 active identity mismatch after peer OFF'; echo 'rank0 foreign/stale identity' >&2; exit 4
  fi
  if [[ "$ls" == ACTIVE ]]; then
    systemctl --user stop ds41-rank0.service
    lp=$(safe_probe_local ds41-rank0.service); ls=$(probe_state "$lp")
  fi
  [[ "$ls" == OFF_VERIFIED ]] || { persist_unreconciled "rank0 stop not verified: $ls"; echo 'rank0 not proven OFF' >&2; exit 4; }

  atomic_owner NONE OFF '' 'both DS41 units/cgroups verified OFF' '' '' '' ''
  echo 'DS41_OFF_VERIFIED'
}

cmd_reconcile() {
  acquire_control
  peer_identity || { persist_unreconciled 'reconcile failed: NODE02 identity unavailable'; echo 'NODE02 UNKNOWN' >&2; exit 3; }
  local lp rp ls rs
  lp=$(safe_probe_local ds41-rank0.service); rp=$(safe_probe_peer ds41-rank1.service)
  ls=$(probe_state "$lp"); rs=$(probe_state "$rp")
  if [[ "$ls" == OFF_VERIFIED && "$rs" == OFF_VERIFIED ]]; then
    atomic_owner NONE OFF '' 'explicit reconciliation: both DS41 units/cgroups verified OFF' '' '' '' ''
    echo 'CLUSTER_OFF_RECONCILED'
    return 0
  fi
  persist_unreconciled "reconcile refused: rank0=$ls rank1=$rs"
  echo "UNRECONCILED rank0=$ls rank1=$rs" >&2
  exit 4
}

cmd_status() {
  local lp rp
  lp=$(safe_probe_local ds41-rank0.service)
  rp=$(safe_probe_peer ds41-rank1.service)
  echo "NODE01_STATE=$(probe_state "$lp")"
  probe_body "$lp"
  echo "NODE02_STATE=$(probe_state "$rp")"
  probe_body "$rp"
  echo 'OWNER'
  cat "$OWNER_FILE" 2>/dev/null || echo '{}'
}

action=${1:?start|stop|status|reconcile}
case "$action" in
  start) cmd_start "${2:?epoch}" "${3:-18210}" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  reconcile) cmd_reconcile ;;
  *) exit 2 ;;
esac
