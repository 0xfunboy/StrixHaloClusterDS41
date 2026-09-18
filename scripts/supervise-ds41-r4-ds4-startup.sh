#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
RAW=$ROOT/reports/DS41-Q2-001/recovery-upstream/r4
REG=$RAW/ds4-lifecycle-registry.json
REL=/home/funboy/.local/share/haloclu-ds41/releases/k2-prefill-5bdfed6
mkdir -p "$RAW"
write(){ python3 - "$REG" "$1" "$2" <<'PY'
import json,os,sys,time
p,state,detail=sys.argv[1:]
o={"schema":"ds41-r4-ds4-lifecycle-v1","state":state,"detail":detail,"updated_unix":time.time(),"coordinator_unit":"ds41-r4-ds4-coordinator.service","worker_unit":"ds41-r4-ds4-worker.service","api":"http://127.0.0.1:8080","tp":"10.55.0.1:9911","k2_rollback":"k2-prefill-5bdfed6"}
t=p+'.tmp'; open(t,'w').write(json.dumps(o,indent=2)+'\n'); os.replace(t,p)
PY
}
cleanup_restore(){
  systemctl --user stop ds41-r4-ds4-coordinator.service 2>/dev/null || true
  ssh -o IdentityAgent=none -o BatchMode=yes 02-evo-x3-tb 'systemctl --user stop ds41-r4-ds4-worker.service 2>/dev/null || true' || true
  sleep 2
  if DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" on >"$RAW/k2-restore-after-ds4.log" 2>&1; then
    write FAILED_RESTORED_K2 "$1"
  else
    write FAILED_RESTORE_K2 "$1; K2 restore failed, inspect k2-restore-after-ds4.log"
  fi
}
write STARTING 'K2 OFF verified; DS4 coordinator/worker launched; waiting for local API readiness'
deadline=$((SECONDS+1200))
while (( SECONDS < deadline )); do
  lc=$(systemctl --user is-active ds41-r4-ds4-coordinator.service 2>/dev/null || true)
  rw=$(ssh -o IdentityAgent=none -o BatchMode=yes 02-evo-x3-tb 'systemctl --user is-active ds41-r4-ds4-worker.service 2>/dev/null || true' 2>/dev/null || true)
  if [[ "$lc" == failed || "$lc" == inactive ]]; then cleanup_restore "coordinator terminal before readiness: $lc"; exit 20; fi
  if [[ "$rw" == failed || "$rw" == inactive ]]; then cleanup_restore "worker terminal before readiness: $rw"; exit 21; fi
  code=$(curl --noproxy '*' -sS -o "$RAW/v1-models.json.tmp" --max-time 3 -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null || true)
  if [[ "$code" == 200 ]]; then
    mv "$RAW/v1-models.json.tmp" "$RAW/v1-models.json"
    write READY 'DS4 coordinator API /v1/models HTTP200; both systemd units active; K2 remains OFF'
    exit 0
  fi
  rm -f "$RAW/v1-models.json.tmp"
  sleep 5
done
cleanup_restore 'DS4 readiness timeout 1200s'
exit 22
