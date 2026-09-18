#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
RAW=$ROOT/reports/DS41-Q2-001/ds4-usable-release-001
REL=/home/funboy/.local/share/haloclu-ds41/releases/k2-prefill-5bdfed6
PEER=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)
status(){
 localc=$(systemctl --user is-active ds4-usable-coordinator.service 2>/dev/null || true)
 remotec=$("${PEER[@]}" 'systemctl --user is-active ds4-usable-worker.service 2>/dev/null || true' 2>/dev/null || true)
 api=$(curl --noproxy '*' -sS -o /dev/null --max-time 3 -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null || true); api=${api:-000}
 k2=$(DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" status)
 state=ERROR; reason=''
 if [[ "$localc" == active && "$remotec" == active && "$api" == 200 ]]; then state=READY
 elif [[ "$localc" == inactive && "$remotec" == inactive ]]; then state=OFF
 elif [[ "$localc" == active && "$remotec" == active ]]; then state=STARTING; reason='both DS4 units active; API not ready'
 else reason="partial DS4 state coordinator=$localc worker=$remotec api=$api"
 fi
 jq -n --arg state "$state" --arg reason "$reason" --arg local "$localc" --arg remote "$remotec" --arg api "$api" --argjson k2 "$k2" '{state:$state,reason:$reason,profile:"DS4_USABLE_RELEASE_001",backend:"DS4",model:"DeepSeek-V4.1-Flash-Q2",target_only:true,dspark:false,context_limit:16384,coordinator:$local,worker:$remote,api_http:$api,k2:$k2}'
}
on(){
 st=$(status); l=$(jq -r .coordinator<<<"$st"); r=$(jq -r .worker<<<"$st"); a=$(jq -r .api_http<<<"$st")
 if [[ "$l" == active && "$r" == active && "$a" == 200 ]]; then echo "$st"; return 0; fi
 [[ "$l" != active && "$r" != active ]] || { echo 'partial DS4 state; refuse start' >&2; echo "$st" >&2; exit 5; }
 kstate=$(jq -r '.k2.state'<<<"$st")
 if [[ "$kstate" == READY ]]; then DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" off >"$RAW/k2-off-before-usable.json"
 elif [[ "$kstate" != OFF ]]; then echo "K2 state $kstate not startable" >&2; exit 6
 fi
 systemctl --user reset-failed ds4-usable-coordinator.service 2>/dev/null || true
 systemd-run --user --unit=ds4-usable-coordinator --collect --property=KillMode=control-group --property=Restart=no --property=TimeoutStopSec=30 --property="StandardOutput=append:$RAW/coordinator.log" --property="StandardError=append:$RAW/coordinator.log" "$ROOT/scripts/run-ds4-usable-node.sh" coordinator
 "${PEER[@]}" "systemctl --user reset-failed ds4-usable-worker.service 2>/dev/null || true; systemd-run --user --unit=ds4-usable-worker --collect --property=KillMode=control-group --property=Restart=no --property=TimeoutStopSec=30 --property='StandardOutput=append:$RAW/worker.log' --property='StandardError=append:$RAW/worker.log' '$ROOT/scripts/run-ds4-usable-node.sh' worker"
 systemctl --user reset-failed ds4-usable-startup-supervisor.service 2>/dev/null || true
 systemd-run --user --unit=ds4-usable-startup-supervisor --collect --property=Restart=no --property=RuntimeMaxSec=1300 --property="StandardOutput=append:$RAW/startup-supervisor.log" --property="StandardError=append:$RAW/startup-supervisor.log" "$ROOT/scripts/supervise-ds4-usable-startup.sh"
 status
}
off(){
 systemctl --user stop ds4-usable-coordinator.service 2>/dev/null || true
 "${PEER[@]}" 'systemctl --user stop ds4-usable-worker.service 2>/dev/null || true' || true
 sleep 2
 status
}
rollback(){
 off >/dev/null || true
 DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" on
}
case ${1:-} in status) status;; on) on;; off) off;; rollback-k2) rollback;; *) echo 'usage: ds4-usable-controller.sh status|on|off|rollback-k2' >&2; exit 2;; esac
