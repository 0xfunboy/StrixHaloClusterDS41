# DS41 TRANSFER DS4 → NATIVE 001 — L0

**PASS.** The native renderer now has an opt-in `ds4-low-v1` profile. It preserves thinking ON and LOW semantics while suppressing only the synthetic numeric `Reasoning Effort: 25 (...)` prefix. Existing NONE/LOW/HIGH behavior is unchanged when the profile is absent.

Evidence:
- actual `DeepseekV4Renderer` API path → candidate tokenizer → **7/7 full token-ID sequences and rendered text exactly equal to DS4** (user-only, explicit system, multi-turn, code2k, docs2k, holdout A/B);
- 5 historical profile fixtures remain token/text identical to the pre-patch native tokenizer;
- incompatible OFF/NONE/HIGH/unknown/content-list/tools requests fail closed;
- canonical request flow uses one `prefill_token_ids` array; target model ingests it once and DSpark inherits DFlash `propose(input_batch, ...)` without tokenizer/chat templating;
- fresh vLLM pin + existing 25-file MMQ/Engram/K2 overlay + new two-file patch reproduces **27/27 candidate files byte-for-byte**.

Patch: `runtime/ds41/patches/vllm-dsv41-ds4-low-v1.patch` (2 files only). L1 release/runner/controller are prepared before model load.
