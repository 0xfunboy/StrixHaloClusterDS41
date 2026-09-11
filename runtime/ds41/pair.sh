#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
ssh_cmd=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)
[[ $(hostname) == 01-EVO-X3 ]] || exit 2
[[ $("${ssh_cmd[@]}" hostname) == 02-EVO-X3 ]] || exit 2
action=${1:?start|stop|status}
case "$action" in
  start)
    epoch=${2:?epoch}; port=${3:-18210}
    ! systemctl --user is-active --quiet ds41-rank0.service || { echo rank0-active >&2; exit 2; }
    ! "${ssh_cmd[@]}" systemctl --user is-active --quiet ds41-rank1.service || { echo rank1-active >&2; exit 2; }
    "${ssh_cmd[@]}" systemd-run --user --unit=ds41-rank1 --collect bash "$ROOT/runtime/ds41/launch-node.sh" 1 10.55.0.2 "$epoch" "$port"
    systemd-run --user --unit=ds41-rank0 --collect bash "$ROOT/runtime/ds41/launch-node.sh" 0 10.55.0.1 "$epoch" "$port"
    ;;
  stop)
    systemctl --user stop ds41-rank0.service 2>/dev/null || true
    "${ssh_cmd[@]}" systemctl --user stop ds41-rank1.service 2>/dev/null || true
    ;;
  status)
    systemctl --user show ds41-rank0.service -p ActiveState -p SubState -p InvocationID 2>/dev/null || true
    "${ssh_cmd[@]}" systemctl --user show ds41-rank1.service -p ActiveState -p SubState -p InvocationID 2>/dev/null || true
    ;;
  *) exit 2;;
esac
