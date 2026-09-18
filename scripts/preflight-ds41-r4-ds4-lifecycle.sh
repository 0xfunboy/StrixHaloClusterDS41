#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/funboy/StrixHaloClusterDS41
DS4=$ROOT/.vendor/ds4-v41-halo-7d0454b
MODEL=/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf
PEER=02-evo-x3-tb
EXPECT_SIZE=365713686528
EXPECT_DS4=11c35f37644484b0beda516c6143edb645acf8ebe53511144bbab2e2cceddf3c
EXPECT_SERVER=d58bde5105a91c531f1099f9948472d885c281c1f80c21a2918e70b64f8d6dcd
EXPECT_BENCH=50902b921bbed64460cff0f443f4d84b90dcb50e6db33cae66421c6cd797c867
REL=/home/funboy/.local/share/haloclu-ds41/releases/k2-prefill-5bdfed6
sha(){ sha256sum "$1" | awk '{print $1}'; }
need_sha(){ local got; got=$(sha "$1"); [[ "$got" == "$2" ]] || { echo "FAIL_SHA $1 got=$got expected=$2"; exit 10; }; }
need_sha "$DS4/ds4" "$EXPECT_DS4"
need_sha "$DS4/ds4-server" "$EXPECT_SERVER"
need_sha "$DS4/ds4-bench" "$EXPECT_BENCH"
remote=$(ssh -o IdentityAgent=none -o BatchMode=yes "$PEER" "sha256sum '$DS4/ds4' '$DS4/ds4-server' '$DS4/ds4-bench'")
grep -q "$EXPECT_DS4" <<<"$remote" || { echo FAIL_PEER_DS4_SHA; exit 11; }
grep -q "$EXPECT_SERVER" <<<"$remote" || { echo FAIL_PEER_SERVER_SHA; exit 12; }
grep -q "$EXPECT_BENCH" <<<"$remote" || { echo FAIL_PEER_BENCH_SHA; exit 13; }
ping -c1 -W1 10.55.0.2 >/dev/null || { echo FAIL_USB4_PING; exit 14; }
python3 - <<'PYPORT'
import socket
for host,port in [('10.55.0.1',9911),('127.0.0.1',8080)]:
    s=socket.socket()
    try: s.bind((host,port))
    except OSError as e: raise SystemExit(f"FAIL_PORT_BUSY node01 {host}:{port} {e}")
    finally: s.close()
PYPORT
ssh -o IdentityAgent=none -o BatchMode=yes "$PEER" "python3 - <<'PYPORT'
import socket
for host,port in [('10.55.0.2',9911),('127.0.0.1',8080)]:
    s=socket.socket()
    try: s.bind((host,port))
    except OSError as e: raise SystemExit(f'FAIL_PORT_BUSY node02 {host}:{port} {e}')
    finally: s.close()
PYPORT"
local_size=0; [[ -f "$MODEL" ]] && local_size=$(stat -c %s "$MODEL")
peer_size=$(ssh -o IdentityAgent=none -o BatchMode=yes "$PEER" "[ -f '$MODEL' ] && stat -c %s '$MODEL' || echo 0")
status=$(DS41_ROOT="$REL" "$REL/runtime/ds41/serve-controller.sh" status)
echo "$status" | jq -e '.state=="READY" and .owner=="DS41" and .epoch=="1789651304744356945"' >/dev/null || { echo FAIL_K2_LIVE_STATE; echo "$status"; exit 17; }
if [[ "$peer_size" != "$EXPECT_SIZE" ]]; then echo "BLOCKED_MODEL_NODE02 size=$peer_size expected=$EXPECT_SIZE"; exit 20; fi
if [[ "$local_size" != "$EXPECT_SIZE" ]]; then
  echo "BLOCKED_MODEL_MIRROR_NODE01 size=$local_size expected=$EXPECT_SIZE k2=READY usb4=PASS binaries=PASS ports=PASS"
  exit 21
fi
echo "PASS_READY_FOR_DS4_LIFECYCLE model_size=$EXPECT_SIZE k2=READY usb4=PASS binaries=PASS ports=PASS"
