#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PAIR="$ROOT/runtime/ds41/pair.sh"
TMP=$(mktemp -d "$ROOT/.test-ds41-control.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/state" "$TMP/root"

cat >"$TMP/bin/systemctl" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
base=${DS41_TEST_STATE:?}
state_file="$base/local.state"; inv_file="$base/local.inv"; env_file="$base/local.env"
state=$(cat "$state_file" 2>/dev/null || echo OFF)
cmd=" $* "
if [[ "$cmd" == *" show "* ]]; then
  [[ "$state" != UNKNOWN ]] || exit 1
  if [[ "$state" == ACTIVE ]]; then
    echo LoadState=loaded; echo ActiveState=active; echo SubState=running; echo MainPID=1111
    echo InvocationID="$(cat "$inv_file")"; echo ControlGroup=/user.slice/mock-local; echo Environment="$(cat "$env_file")"
  else
    echo LoadState=loaded; echo ActiveState=inactive; echo SubState=dead; echo MainPID=0
    echo InvocationID="$(cat "$inv_file" 2>/dev/null || true)"; echo ControlGroup=; echo Environment=
  fi
  exit 0
fi
if [[ "$cmd" == *" stop "* ]]; then echo OFF >"$state_file"; exit 0; fi
if [[ "$cmd" == *" reset-failed "* ]]; then exit 0; fi
exit 0
SH

cat >"$TMP/bin/systemd-run" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
base=${DS41_TEST_STATE:?}
epoch= nonce=
for a in "$@"; do
  case "$a" in
    --setenv=DS41_OWNER_EPOCH=*) epoch=${a#*=DS41_OWNER_EPOCH=} ;;
    --setenv=DS41_OWNER_NONCE=*) nonce=${a#*=DS41_OWNER_NONCE=} ;;
  esac
done
[[ -n "$epoch" && -n "$nonce" ]]
echo ACTIVE >"$base/local.state"
printf '%032d\n' 11 >"$base/local.inv"
printf 'DS41_OWNER_EPOCH=%s DS41_OWNER_NONCE=%s\n' "$epoch" "$nonce" >"$base/local.env"
echo $(( $(cat "$base/local.starts" 2>/dev/null || echo 0) + 1 )) >"$base/local.starts"
echo 'Running as unit: ds41-rank0.service'
SH

cat >"$TMP/peer" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
base=${DS41_TEST_STATE:?}; mode=$(cat "$base/peer.mode" 2>/dev/null || echo normal)
if [[ "$mode" == ssh_unknown || -e "$base/peer.lose" ]]; then exit 255; fi
if [[ "${1:-}" == hostname ]]; then echo 02-EVO-X3; exit 0; fi
if [[ "${1:-}" == /bin/bash ]]; then
  state=$(cat "$base/peer.state" 2>/dev/null || echo OFF)
  [[ "$state" != UNKNOWN ]] || exit 255
  if [[ "$state" == ACTIVE ]]; then
    echo LoadState=loaded; echo ActiveState=active; echo SubState=running; echo MainPID=2222
    echo InvocationID="$(cat "$base/peer.inv")"; echo ControlGroup=/user.slice/mock-peer; echo Environment="$(cat "$base/peer.env")"; echo CgroupPIDs=
  else
    echo LoadState=loaded; echo ActiveState=inactive; echo SubState=dead; echo MainPID=0
    echo InvocationID="$(cat "$base/peer.inv" 2>/dev/null || true)"; echo ControlGroup=; echo Environment=; echo CgroupPIDs=
  fi
  exit 0
fi
cmd=" $* "
if [[ "$cmd" == *" systemctl --user stop ds41-rank1.service "* ]]; then
  echo OFF >"$base/peer.state"
  [[ "$mode" != loss_after_stop ]] || touch "$base/peer.lose"
  exit 0
fi
if [[ "$cmd" == *" systemctl --user reset-failed ds41-rank1.service "* ]]; then exit 0; fi
if [[ "${1:-}" == systemd-run ]]; then
  epoch= nonce=
  for a in "$@"; do
    case "$a" in
      --setenv=DS41_OWNER_EPOCH=*) epoch=${a#*=DS41_OWNER_EPOCH=} ;;
      --setenv=DS41_OWNER_NONCE=*) nonce=${a#*=DS41_OWNER_NONCE=} ;;
    esac
  done
  [[ -n "$epoch" && -n "$nonce" ]]
  echo ACTIVE >"$base/peer.state"
  printf '%032d\n' 22 >"$base/peer.inv"
  printf 'DS41_OWNER_EPOCH=%s DS41_OWNER_NONCE=%s\n' "$epoch" "$nonce" >"$base/peer.env"
  echo $(( $(cat "$base/peer.starts" 2>/dev/null || echo 0) + 1 )) >"$base/peer.starts"
  echo 'Running as unit: ds41-rank1.service'
  exit 0
fi
exit 0
SH
chmod +x "$TMP/bin/systemctl" "$TMP/bin/systemd-run" "$TMP/peer"

export DS41_TEST_STATE="$TMP/state"
export DS41_TEST_PEER_CMD="$TMP/peer"
export DS41_SHARED_STATE="$TMP/shared"
export DS41_ROOT="$TMP/root"
export PATH="$TMP/bin:/usr/bin:/bin"
mkdir -p "$DS41_SHARED_STATE"

write_owner() {
  local owner=$1 state=$2 epoch=$3 inv0=$4 nonce0=$5 inv1=$6 nonce1=$7
  jq -n --arg owner "$owner" --arg state "$state" --arg epoch "$epoch" \
    --arg i0 "$inv0" --arg n0 "$nonce0" --arg i1 "$inv1" --arg n1 "$nonce1" \
    '{schema:"strix-cluster-compute-v2",owner:$owner,state:$state,epoch:$epoch,detail:"test",updated:"test",
      ranks:{"0":{unit:"ds41-rank0.service",invocation_id:$i0,nonce:$n0},"1":{unit:"ds41-rank1.service",invocation_id:$i1,nonce:$n1}}}' >"$DS41_SHARED_STATE/owner.json"
}
set_unit() {
  local which=$1 state=$2 epoch=${3:-} nonce=${4:-} inv=${5:-}
  echo "$state" >"$TMP/state/$which.state"
  printf '%s\n' "$inv" >"$TMP/state/$which.inv"
  printf 'DS41_OWNER_EPOCH=%s DS41_OWNER_NONCE=%s\n' "$epoch" "$nonce" >"$TMP/state/$which.env"
}
reset_case() {
  rm -rf "$TMP/state" "$DS41_SHARED_STATE" "$TMP/root/reports"; mkdir -p "$TMP/state" "$DS41_SHARED_STATE"
  echo normal >"$TMP/state/peer.mode"
}
expect_fail() {
  set +e; "$@" >/tmp/ds41-control-test.out 2>&1; rc=$?; set -e
  (( rc != 0 )) || { cat /tmp/ds41-control-test.out; echo 'expected failure' >&2; exit 1; }
}

# 1) Normal start, duplicate start, stop.
reset_case
set_unit local OFF; set_unit peer OFF
write_owner NONE OFF '' '' '' '' ''
"$PAIR" start 1789124001 18210 >/dev/null
[[ $(jq -r .state "$DS41_SHARED_STATE/owner.json") == RUNNING ]]
[[ $(cat "$TMP/state/local.starts") == 1 && $(cat "$TMP/state/peer.starts") == 1 ]]
"$PAIR" start 1789124999 18210 | grep -q DS41_ALREADY_RUNNING
[[ $(cat "$TMP/state/local.starts") == 1 && $(cat "$TMP/state/peer.starts") == 1 ]]
"$PAIR" stop | grep -q DS41_OFF_VERIFIED
[[ $(jq -r '.owner+":"+.state' "$DS41_SHARED_STATE/owner.json") == NONE:OFF ]]

# 2) SSH loss after peer stop: peer becomes UNKNOWN and local rank must remain active.
reset_case
n0=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; n1=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
i0=11111111111111111111111111111111; i1=22222222222222222222222222222222; ep=1789124002
set_unit local ACTIVE "$ep" "$n0" "$i0"; set_unit peer ACTIVE "$ep" "$n1" "$i1"
write_owner DS41 RUNNING "$ep" "$i0" "$n0" "$i1" "$n1"
echo loss_after_stop >"$TMP/state/peer.mode"
expect_fail "$PAIR" stop
[[ $(cat "$TMP/state/local.state") == ACTIVE ]]
[[ $(jq -r .state "$DS41_SHARED_STATE/owner.json") == UNRECONCILED ]]

# 3) Rank0 died and peer cannot be verified: flock is free but new start remains blocked.
reset_case
set_unit local OFF
echo UNKNOWN >"$TMP/state/peer.state"; echo ssh_unknown >"$TMP/state/peer.mode"
write_owner DS41 RUNNING 1789124003 "$i0" "$n0" "$i1" "$n1"
expect_fail "$PAIR" start 1789125003 18210
[[ $(jq -r .state "$DS41_SHARED_STATE/owner.json") == UNRECONCILED ]]
[[ ! -e "$TMP/state/local.starts" ]]

# 4) Stale previous-epoch receipt blocks a new start even when both units are OFF.
reset_case
set_unit local OFF; set_unit peer OFF
write_owner DS41 RUNNING 1789000000 "$i0" "$n0" "$i1" "$n1"
expect_fail "$PAIR" start 1789125004 18210
[[ $(jq -r .state "$DS41_SHARED_STATE/owner.json") == UNRECONCILED ]]
[[ ! -e "$TMP/state/local.starts" && ! -e "$TMP/state/peer.starts" ]]
# Explicit reconciliation is allowed only because both probes succeed as OFF_VERIFIED.
"$PAIR" reconcile | grep -q CLUSTER_OFF_RECONCILED
[[ $(jq -r '.owner+":"+.state' "$DS41_SHARED_STATE/owner.json") == NONE:OFF ]]

echo 'DS41_CONTROL_TESTS=PASS'
