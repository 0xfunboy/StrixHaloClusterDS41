# DS41-Q2-001 attempt037 — real DSpark asset gate

**Status:** `BLOCKED_MISSING_DSPARK_WEIGHTS / NO_MODEL_LOAD / NO_WEIGHT_DOWNLOAD`

## DSpark support

Pinned vLLM supports DeepSeek V4.1 DSpark through `DSparkV41DraftModel`. Model-free config admission with the current target passes for **K=1**: `method=dspark`, `parallel_drafting=true`, `num_speculative_tokens=1`, `n_predict=5`, three learned DSpark stages, target hidden layers `[37,38,39]`, draft TP2. K=1 therefore does **not** remove or bypass any trained MTP stage.

## Missing local assets

The current DenseFix GGUF contains **0** `mtp`/`dspark`/`nextn` tensors. Its own pack receipt records 67 DSpark/nextn tensors with no GGUF slot, 5 DSpark-only tensors, and 1152 unresolved DSpark/nextn expert tensors. The pinned DSpark loader expects `mtp.{0,1,2}.*`.

Official source: `deepseek-ai/DeepSeek-V4.1-Flash` revision `2bc89ac599031fa673cab993f1df02fc4a98c673`. All 2401 required `mtp.*` tensors are isolated in exactly three MTP-only source shards:

| file | mtp tensors | bytes | GiB | SHA256 |
|---|---:|---:|---:|---|
| `model-00044-of-00048.safetensors` | 801 | 2,652,728,736 | 2.471 | `9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05` |
| `model-00045-of-00048.safetensors` | 798 | 2,573,998,176 | 2.397 | `0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5` |
| `model-00046-of-00048.safetensors` | 802 | 2,706,402,896 | 2.521 | `e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d` |

Whole-shard acquisition: **7,933,129,808 bytes (7.388 GiB)**. None of these files exists under the local model/HF caches. No checkpoint payload was downloaded during this audit; only the index/API metadata and ~85 KB Safetensors headers per shard were read.

## Native format and memory

These are native source weights, not GGUF: FP4 expert payload (packed `I8`) with `F8_E8M0` scales, FP8 dense/attention plus BF16/F32 tensors. Source quantization contract is FP8 dynamic, block `[32,32]`, UE8M0, expert dtype FP4.

Tensor payload is **7,932,874,632 bytes** total, of which **7,219,445,760 bytes** are routed-expert payload. Under TP2/EP2 the parameter-only residency is bounded at roughly **3.69–4.03 GiB/rank**, before draft KV/workspace/allocator overhead.

## Loader blocker

The pin exposes `SpeculativeConfig.draft_load_config`, but `load_dspark_model()` currently calls `get_model()` without passing that draft-specific load config, and `get_draft_quant_config()` reads the target/global `vllm_config.load_config`. With the GGUF target, the K=1 config audit consequently resolves the draft quantization as `gguf`. That is not valid for these native FP8/FP4 MTP shards.

Minimum safe integration after weight approval:

1. Acquire only shards 44–46 above and verify their exact SHA256.
2. Create a local DSpark sidecar metadata directory with the source config and a filtered Safetensors index containing only the 2401 `mtp.*` entries.
3. Point the speculative model at that sidecar and use a draft-specific Safetensors/auto load config while the target remains GGUF.
4. Patch/test the pinned DSpark loader and draft quant resolution to honor `draft_load_config`; retain the source FP8/FP4 quant contract.
5. Only then perform the first real K=1 drafter load/functional run.

Target embedding and LM head are shareable by the pinned loader; `mtp.2.markov_head.embed/head` are learned DSpark factors and remain required.

## Decision

**STOP at the asset gate.** The requested real DSpark measurement cannot be performed honestly with the assets currently local. Acquiring the three missing MTP shards requires separate owner confirmation. No target/model load, DSpark inference, driver/dependency change, or weight download was performed.
