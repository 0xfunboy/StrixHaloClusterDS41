# DS41-Q2-001 attempt038 — MTP acquisition terminal

**Status:** `PASS / EXACT_REVISION / BOTH_NODES_VERIFIED`.

Downloaded once on NODE01 from `deepseek-ai/DeepSeek-V4.1-Flash@2bc89ac599031fa673cab993f1df02fc4a98c673` using resume, then replicated over `02-evo-x3-tb`. Exactly three authorized MTP-only shards, **7,933,129,808 bytes**, passed frozen size+SHA256 on both nodes. No other weight payload was downloaded.

Sidecar `/home/funboy/models/ds41/dspark-v41-mtp-2bc89ac` contains source `config.json` plus a filtered Safetensors index with exactly **2401 `mtp.*` tensors** referencing shards44–46. No weights are stored in Git.
