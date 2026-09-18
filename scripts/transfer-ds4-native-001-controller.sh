#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/worktrees/ds41-transfer-ds4-native-001
RELEASE=/home/funboy/.local/share/haloclu-ds41/releases/native-ds4low-k2-transfer001
DS4=/home/funboy/StrixHaloClusterDS41/scripts/ds4-document-controller.sh
case ${1:-} in
  status)
    native=$("$RELEASE/runtime/ds41/serve-controller.sh" status 2>/dev/null || echo '{"state":"UNKNOWN"}')
    ds4=$("$DS4" status 2>/dev/null || echo '{"state":"UNKNOWN"}')
    jq -n --argjson native "$native" --argjson ds4 "$ds4" '{profile:"DS41_TRANSFER_DS4_NATIVE_001",native:$native,ds4:$ds4}'
    ;;
  native-on)
    d=$("$DS4" status | jq -r .state)
    if [[ "$d" == READY ]]; then "$DS4" off >/dev/null; elif [[ "$d" != OFF ]]; then echo "DS4 state $d blocks native start" >&2; exit 4; fi
    "$RELEASE/runtime/ds41/serve-controller.sh" on
    ;;
  native-off)
    "$RELEASE/runtime/ds41/serve-controller.sh" off
    ;;
  fallback-ds4)
    "$RELEASE/runtime/ds41/serve-controller.sh" off >/dev/null 2>&1 || true
    "$DS4" on
    ;;
  *) echo 'usage: transfer-ds4-native-001-controller.sh status|native-on|native-off|fallback-ds4' >&2; exit 2;;
esac
