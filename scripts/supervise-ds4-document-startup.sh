#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
RAW=$ROOT/reports/DS41-Q2-001/ds4-document-profile-002
REG=$RAW/lifecycle-registry.json
REL=/home/funboy/.local/share/haloclu-ds41/releases/k2-prefill-5bdfed6
OWNER=DS4_DOCUMENT_PROFILE_002_20260918
mkdir -p "$RAW"
write(){
 python3 - "$REG" "$1" "$2" "$OWNER" <<'PY'
import json,os,sys,time
p,state,detail,owner=sys.argv[1:]
o={'schema':'ds4-document-profile-002-lifecycle-v1','owner':owner,'state':state,'detail':detail,'updated_unix':time.time(),'coordinator_unit':'ds4-document-coordinator.service','worker_unit':'ds4-document-worker.service','api':'http://127.0.0.1:8080','tp':'10.55.0.1:9911','gate_timeout_ms':5000,'ctx':16384,'rollback':'k2-prefill-5bdfed6'}
t=p+'.tmp';open(t,'w').write(json.dumps(o,indent=2)+'\n');os.replace(t,p)
PY
}
cleanup_restore(){
 systemctl --user stop ds4-document-coordinator.service 2>/dev/null || true
 ssh -o IdentityAgent=none -o BatchMode=yes 02-evo-x3-tb 'systemctl --user stop ds4-document-worker.service 2>/dev/null || true' || true
 sleep 2
 if DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" on >"$RAW/startup-k2-rollback.log" 2>&1; then
   write FAILED_RESTORED_K2 "$1"
 else
   write FAILED_K2_RESTORE "$1; inspect startup-k2-rollback.log"
 fi
}
write STARTING 'DOCUMENT_PROFILE_002 launched; K2 OFF verified; waiting DS4 API'
deadline=$((SECONDS+1200))
while ((SECONDS<deadline)); do
 lc=$(systemctl --user is-active ds4-document-coordinator.service 2>/dev/null || true)
 rw=$(ssh -o IdentityAgent=none -o BatchMode=yes 02-evo-x3-tb 'systemctl --user is-active ds4-document-worker.service 2>/dev/null || true' 2>/dev/null || true)
 if [[ "$lc" == failed || "$lc" == inactive ]]; then cleanup_restore "coordinator terminal before readiness: $lc"; exit 20; fi
 if [[ "$rw" == failed || "$rw" == inactive ]]; then cleanup_restore "worker terminal before readiness: $rw"; exit 21; fi
 code=$(curl --noproxy '*' -sS -o "$RAW/v1-models.tmp" --max-time 3 -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null || true)
 if [[ "$code" == 200 ]]; then mv "$RAW/v1-models.tmp" "$RAW/v1-models.json"; write READY 'DS4 API HTTP200; both units active; K2 OFF'; exit 0; fi
 rm -f "$RAW/v1-models.tmp"; sleep 5
done
cleanup_restore 'readiness timeout 1200s'; exit 22
