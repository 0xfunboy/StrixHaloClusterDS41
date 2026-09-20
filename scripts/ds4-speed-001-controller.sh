#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/funboy/StrixHaloClusterDS41
RAW=/home/funboy/reports/DS4-SPEED-001
QUALIFIED=$ROOT/scripts/ds4-document-controller.sh
NODE=$ROOT/scripts/run-ds4-speed-001-node.sh
PEER=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)

speed_status() {
    local local_state remote_state api mode=none timing=false local_pid=0
    local_state=$(systemctl --user is-active ds4-speed-001-coordinator.service 2>/dev/null || true)
    remote_state=$("${PEER[@]}" 'systemctl --user is-active ds4-speed-001-worker.service 2>/dev/null || true' 2>/dev/null || true)
    api=$(curl --noproxy '*' -sS -o /dev/null --max-time 3 -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null || true)
    api=${api:-000}
    if [[ "$local_state" == active ]]; then
        local_pid=$(systemctl --user show ds4-speed-001-coordinator.service -p MainPID --value)
        if tr '\0' '\n' <"/proc/$local_pid/environ" | grep -qx 'DS4_V41_DISABLE_ENGRAM_CONCURRENT=1'; then
            mode=serial-rollback
        else
            mode=concurrent
        fi
        if tr '\0' '\n' <"/proc/$local_pid/environ" | grep -qx 'DS4_V41_ENGRAM_TIMING=1'; then
            timing=true
        fi
    fi
    local state=ERROR
    if [[ "$local_state" == active && "$remote_state" == active && "$api" == 200 ]]; then state=READY
    elif [[ "$local_state" == inactive && "$remote_state" == inactive ]]; then state=OFF
    elif [[ "$local_state" == active && "$remote_state" == active ]]; then state=STARTING
    fi
    jq -n --arg state "$state" --arg local "$local_state" --arg remote "$remote_state" \
        --arg api "$api" --arg mode "$mode" --argjson timing "$timing" \
        '{campaign:"DS4_SPEED_001",state:$state,coordinator:$local,worker:$remote,
          api_http:$api,release:"ds4-speed-001-engram1",engram_mode:$mode,
          diagnostic_timing:$timing,verify2:false}'
}

managed_status() {
    local speed qualified
    speed=$(speed_status)
    if [[ $(jq -r .state <<<"$speed") != OFF ]]; then
        echo "$speed"
        return 0
    fi
    qualified=$($QUALIFIED status)
    if [[ $(jq -r .state <<<"$qualified") != OFF ]]; then
        jq -n --argjson q "$qualified" \
            '{campaign:"DS4_SPEED_001",state:$q.state,coordinator:$q.coordinator,
              worker:$q.worker,api_http:$q.api_http,
              release:"ds4-v41-halo-7d0454b",engram_mode:"original-serial",
              diagnostic_timing:false,verify2:false,mode:"qualified-rollback"}'
        return 0
    fi
    echo "$speed"
}

off_pair() {
    systemctl --user stop ds4-speed-001-coordinator.service 2>/dev/null || true
    "${PEER[@]}" 'systemctl --user stop ds4-speed-001-worker.service 2>/dev/null || true' || true
    sleep 2
    speed_status
}

wait_ready() {
    local controller=$1
    local deadline=$((SECONDS + 1300))
    while (( SECONDS < deadline )); do
        local status state
        status=$($controller status)
        state=$(jq -r .state <<<"$status")
        if [[ "$state" == READY ]]; then echo "$status"; return 0; fi
        if [[ "$state" == ERROR ]]; then echo "$status" >&2; return 1; fi
        sleep 5
    done
    $controller status >&2
    return 1
}

start_pair() {
    local arm=${1:?serial|concurrent}
    local tag=${2:?tag}
    local timing=${3:-0}
    [[ "$arm" == serial || "$arm" == concurrent ]] || { echo invalid_arm >&2; exit 2; }
    [[ "$timing" == 0 || "$timing" == 1 ]] || { echo invalid_timing >&2; exit 2; }
    [[ "$tag" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo invalid_tag >&2; exit 2; }

    local speed qualified
    speed=$(speed_status)
    [[ $(jq -r .state <<<"$speed") == OFF ]] || { echo SPEED_PAIR_NOT_OFF >&2; echo "$speed" >&2; exit 5; }
    qualified=$($QUALIFIED status)
    if [[ $(jq -r .state <<<"$qualified") == READY ]]; then
        $QUALIFIED off >"$RAW/qualified-off-before-$tag.json"
    elif [[ $(jq -r .state <<<"$qualified") != OFF ]]; then
        echo QUALIFIED_PAIR_NOT_SAFE >&2
        echo "$qualified" >&2
        exit 6
    fi

    local run=$RAW/runtime/$tag
    mkdir -p "$run"
    "${PEER[@]}" "mkdir -p '$run'"
    systemctl --user reset-failed ds4-speed-001-coordinator.service 2>/dev/null || true
    "${PEER[@]}" 'systemctl --user reset-failed ds4-speed-001-worker.service 2>/dev/null || true'
    "${PEER[@]}" "systemd-run --user --unit=ds4-speed-001-worker --collect --property=KillMode=control-group --property=Restart=no --property=TimeoutStopSec=30 --property='StandardOutput=append:$run/worker.log' --property='StandardError=append:$run/worker.log' '$NODE' worker '$arm' '$timing'" >"$run/worker-start.txt"
    systemd-run --user --unit=ds4-speed-001-coordinator --collect --property=KillMode=control-group --property=Restart=no --property=TimeoutStopSec=30 --property="StandardOutput=append:$run/coordinator.log" --property="StandardError=append:$run/coordinator.log" "$NODE" coordinator "$arm" "$timing" >"$run/coordinator-start.txt"
    if ! wait_ready speed_status | tee "$run/ready.json"; then
        off_pair >"$run/off-after-start-failure.json" || true
        $QUALIFIED on >"$run/restore-after-start-failure.json" || true
        exit 7
    fi
}

restore_qualified() {
    mkdir -p "$RAW"
    off_pair >"$RAW/speed-off-before-restore.json" || true
    local qualified
    qualified=$($QUALIFIED status)
    if [[ $(jq -r .state <<<"$qualified") == READY ]]; then echo "$qualified"; return 0; fi
    [[ $(jq -r .state <<<"$qualified") == OFF ]] || { echo QUALIFIED_PAIR_NOT_OFF >&2; exit 8; }
    $QUALIFIED on >"$RAW/qualified-start.json"
    wait_ready "$QUALIFIED" | tee "$RAW/qualified-restored.json"
}

off_managed() {
    local speed qualified
    speed=$(speed_status)
    if [[ $(jq -r .state <<<"$speed") != OFF ]]; then
        off_pair >/dev/null
    else
        qualified=$($QUALIFIED status)
        if [[ $(jq -r .state <<<"$qualified") != OFF ]]; then
            $QUALIFIED off >/dev/null
        fi
    fi
    managed_status
}

mkdir -p "$RAW"
case ${1:-} in
status) managed_status ;;
on) start_pair concurrent product-e1 0 ;;
start) start_pair "${2:-}" "${3:-}" "${4:-0}" ;;
off) off_managed ;;
restore|rollback-qualified) restore_qualified ;;
*) echo 'usage: ds4-speed-001-controller.sh status|on|start serial|concurrent TAG [TIMING0|1]|off|restore|rollback-qualified' >&2; exit 2 ;;
esac
