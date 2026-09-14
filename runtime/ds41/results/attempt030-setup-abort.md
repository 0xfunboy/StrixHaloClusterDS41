# Attempt030: setup abort before model load

Status: **SETUP_FAIL_BEFORE_MODEL / PEER_PREFLIGHT_SOURCE_IDENTITY**.

The narrow rowwise downstream protocol and prompt were frozen at source `302d16412a47a19e752f90bad5e03d586801c520`. NODE01 artifact `verify-fast` PASSed and the model-free V2 bridge preflight PASSed (`target-only CONFIG_PASS`, `diagnostic-bridge-Kmax3 CONFIG_PASS`, weight-free CPU factory PASS). NODE02 artifact `verify-fast` also PASSed and resolved the synchronized `.source-commit` correctly, but `scripts/preflight-ds41-small-block.py` aborted before config construction because it unconditionally executed `git -C /home/funboy/StrixHaloClusterDS41 rev-parse HEAD`; the NODE02 deployment is intentionally source-marker based and has no `.git` directory.

No `pair.sh start`, LLM construction, model allocation, generation request, tensor capture or benchmark occurred. Both ranks remain `OFF_VERIFIED`, owner `NONE/OFF`; GLM remains OFF. This is not a model/correctness result and does not consume the narrow numerical experiment.

Correction is limited to making the model-free preflight source identity follow the runtime's existing contract: prefer Git HEAD when present, otherwise validate/read `.source-commit`. The original attempt030 raw/preregister and peer error are preserved. Retry uses a new namespace; no unchanged attempt030 replay.
