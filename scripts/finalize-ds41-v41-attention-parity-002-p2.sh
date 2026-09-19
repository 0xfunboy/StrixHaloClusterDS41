#!/usr/bin/env bash
set -Eeuo pipefail
WT=/home/funboy/worktrees/ds41-v41-attention-parity-002
RUN_ID=${DS41_ATTN002_RUN_ID:-p2-m1}
OUT="$WT/reports/DS41-Q2-001/attention-parity-002/$RUN_ID"
TERM="$OUT/terminal.json"
CTL="$WT/scripts/attention-parity-002-controller.sh"
RAW=${DS41_ATTN002_RAW:-/home/funboy/reports/DS41-ATTENTION-PARITY-002}
mkdir -p "$RAW"
[[ -f "$TERM" ]] || { echo "missing P2 terminal $TERM" >&2; exit 2; }
status=$(jq -r .status "$TERM")
requests=$(jq -r .requests_used "$TERM")
if [[ "$status" == P2_M1_SIX_PASS ]]; then
  final_status=P2_M1_SIX_PASS_LEFT_READY
  ctl=$("$CTL" status)
else
  ctl=$("$CTL" fallback-ds4)
  ds4_state=$(jq -r 'if .ds4 then .ds4.state else .state end // "UNKNOWN"' <<<"$ctl")
  [[ "$ds4_state" == READY ]] || {
    echo "DS4 fallback did not reach READY" >&2
    printf '%s\n' "$ctl" >&2
    exit 5
  }
  final_status="${status}_DS4_READY"
fi
jq -n --arg status "$final_status" --arg terminal "$status" --argjson requests "$requests" --argjson controller "$ctl"   '{schema:"ds41-v41-attention-parity-002-p2-finalizer-v1",status:$status,terminal:$terminal,requests_used:$requests,controller:$controller}'   | tee "$OUT/finalizer.json" "$RAW/p2-finalizer.json"
