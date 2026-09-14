# DS41-Q2-001 attempt038 — draft-native loader gate

**Status:** `PASS / TARGET_GGUF_DRAFT_SAFETENSORS_DEEPSEEK_FP8_INDEPENDENT / NO_MODEL_LOAD`.

Target remains `GGUFConfig` with `load_format=gguf`, `config_format=gguf`. DSpark sidecar resolves `DeepseekV4FP8Config` / `deepseek_v4_fp8` with `load_format=safetensors`, config parser `auto`, K=1, all three stages and target hidden layers37/38/39. The target global load config is not mutated. DeepSeek V4.1 quantization is re-resolved after DSpark restores the real model type, so routed `expert_dtype=fp4` uses the expert-aware quant config rather than generic FP8.
