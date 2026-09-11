#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
SHARED_STATE=/home/funboy/.local/state/strix-cluster
ssh_base=(ssh -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 02-evo-x3-tb)
[[ $(hostname) == 01-EVO-X3 ]] || exit 2
action=${1:?start|stop|status}
peer() { "${ssh_base[@]}" "$@"; }
peer_identity() { [[ $(peer hostname) == 02-EVO-X3 ]]; }
unit_active_local() { systemctl --user is-active --quiet "$1"; }
unit_active_peer() { peer systemctl --user is-active --quiet "$1"; }
case "$action" in
  start)
    epoch=${2:?epoch}; port=${3:-18210}
    peer_identity || { echo 'NODE02 unavailable; refusing partial DS41 start' >&2; exit 3; }
    ! unit_active_local ds41-rank0.service || { echo rank0-active >&2; exit 2; }
    ! unit_active_peer ds41-rank1.service || { echo rank1-active >&2; exit 2; }
    # Rank0 owns the shared compute lock. Start it first and prove that it
    # survived lock acquisition before any model-bearing process starts remotely.
    systemd-run --user --unit=ds41-rank0 --collect bash "$ROOT/runtime/ds41/launch-node.sh" 0 10.55.0.1 "$epoch" "$port"
    for _ in {1..20}; do
      unit_active_local ds41-rank0.service && break
      sleep .25
    done
    unit_active_local ds41-rank0.service || { echo 'rank0 failed before cluster-lock ownership' >&2; exit 4; }
    grep -q '"owner":"DS41"' "$SHARED_STATE/owner.json" 2>/dev/null || { systemctl --user stop ds41-rank0.service || true; echo 'DS41 ownership receipt missing' >&2; exit 4; }
    peer systemd-run --user --unit=ds41-rank1 --collect bash "$ROOT/runtime/ds41/launch-node.sh" 1 10.55.0.2 "$epoch" "$port"
    ;;
  stop)
    # Never release rank0's compute lock while a model-bearing peer is
    # unverifiable. Stop and verify rank1 first, then stop the lock owner.
    peer_identity || { echo 'NODE02 unverifiable; preserving rank0 lock owner' >&2; exit 3; }
    peer systemctl --user stop ds41-rank1.service 2>/dev/null || true
    if unit_active_peer ds41-rank1.service; then
      echo 'rank1 not proven stopped; preserving rank0 lock owner' >&2; exit 4
    fi
    systemctl --user stop ds41-rank0.service 2>/dev/null || true
    unit_active_local ds41-rank0.service && { echo 'rank0 not proven stopped' >&2; exit 4; }
    mkdir -p "$SHARED_STATE"
    exec 9>"$SHARED_STATE/compute.lock"
    if flock -n 9; then
      tmp="$SHARED_STATE/.owner.$$.tmp"
      printf '{"owner":"NONE","state":"OFF","updated":"%s"}\n' "$(date -u +%FT%TZ)" >"$tmp"
      chmod 600 "$tmp"; mv -f "$tmp" "$SHARED_STATE/owner.json"
    fi
    ;;
  status)
    echo 'NODE01'
    systemctl --user show ds41-rank0.service -p ActiveState -p SubState -p InvocationID -p MainPID 2>/dev/null || true
    echo 'NODE02'
    if peer_identity; then
      peer systemctl --user show ds41-rank1.service -p ActiveState -p SubState -p InvocationID -p MainPID 2>/dev/null || true
    else
      echo 'State=UNVERIFIABLE'
    fi
    echo 'OWNER'
    cat "$SHARED_STATE/owner.json" 2>/dev/null || echo '{}'
    ;;
  *) exit 2;;
esac
