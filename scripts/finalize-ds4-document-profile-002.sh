#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
RAW=$ROOT/reports/DS41-Q2-001/ds4-document-profile-002
REG=$RAW/lifecycle-registry.json
OWNER=DS4_DOCUMENT_PROFILE_002_20260918
DECISION=${1:?QUALIFIED|NOT_QUALIFIED}
[[ -f "$REG" ]] || { echo 'missing lifecycle registry' >&2; exit 2; }
got=$(jq -r .owner "$REG")
[[ "$got" == "$OWNER" ]] || { echo "owner mismatch $got" >&2; exit 3; }
case "$DECISION" in
 QUALIFIED)
   "$ROOT/scripts/ds4-document-controller.sh" status | jq -e '.state=="READY" and .owner=="DS4_DOCUMENT_PROFILE_002_20260918"' >/dev/null
   jq -n --arg owner "$OWNER" '{owner:$owner,state:"QUALIFIED_LEFT_READY",rollback_available:"scripts/ds4-document-controller.sh rollback-k2"}' >"$RAW/finalizer.json"
   ;;
 NOT_QUALIFIED)
   "$ROOT/scripts/ds4-document-controller.sh" off >"$RAW/final-ds4-off.json"
   systemctl --user reset-failed ds4-document-final-k2-rollback.service 2>/dev/null || true
   systemd-run --user --unit=ds4-document-final-k2-rollback --collect --property=Restart=no --property=RuntimeMaxSec=1500 --property="StandardOutput=append:$RAW/final-k2-rollback.log" --property="StandardError=append:$RAW/final-k2-rollback.log" /bin/bash -lc 'REL=/home/funboy/.local/share/haloclu-ds41/releases/k2-prefill-5bdfed6; DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" on'
   jq -n --arg owner "$OWNER" '{owner:$owner,state:"ROLLBACK_K2_IN_FLIGHT"}' >"$RAW/finalizer.json"
   ;;
 *) echo invalid_decision >&2; exit 4;;
esac
