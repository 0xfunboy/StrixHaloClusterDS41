#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$ROOT"
bash -n runtime/ds41/pair.sh runtime/ds41/launch-node.sh runtime/ds41/serve-controller.sh
node --check web/app.js
python3 -m py_compile runtime/ds41/artifact_identity.py
jq -e '.model=="DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2" and .lifecycle_preset=="dspark-k2-gfx1151" and .listen=="127.0.0.1:18222" and .backend=="http://127.0.0.1:18221"' config.ds41-k2-serving.json >/dev/null
jq -e '.rank_urls==["http://10.55.0.1:18220","http://10.55.0.2:18220"] and .frontend_listen=="127.0.0.1:18221"' runtime/cluster.ds41-k2.json >/dev/null
jq -e '.preset=="runtime/ds41/presets/dspark-k2-gfx1151.json" and .numerics.dspark_k==2 and .autoload_on_gateway_start==false' runtime/ds41/serving-k2-release.json >/dev/null
status=$(runtime/ds41/serve-controller.sh status)
jq -e '.state=="OFF" and .owner=="NONE" and (.ranks|all(.active==false))' <<<"$status" >/dev/null
grep -q -- '--speculative-config' runtime/ds41/launch-node.sh
grep -q -- '--reasoning-parser deepseek_v41' runtime/ds41/launch-node.sh
grep -q 'num_speculative_tokens.*2' runtime/ds41/launch-node.sh
grep -q 'DS41_RUNTIME_MAX_SEC=infinity' runtime/ds41/serve-controller.sh
printf 'PASS ds41-k2-serving-model-free\n'
