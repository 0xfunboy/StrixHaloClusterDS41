# DS41-Q2-001 attempt038 — draft-native loader gate

**Status:** `PASS / TARGET_GGUF_DRAFT_SAFETENSORS_INDEPENDENT / NO_MODEL_LOAD`.

Target remains `GGUFConfig` with `load_format=gguf`, `config_format=gguf`. DSpark sidecar resolves `Fp8Config` with `load_format=safetensors`, config parser `auto`, K=1, all three stages and target hidden layers37/38/39. The target global load config is not mutated. Full pinned vLLM overlay fresh-applies byte-for-byte from the vendor pin.
