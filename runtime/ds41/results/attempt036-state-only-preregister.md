# DS41-Q2-001 attempt036 — T3 recovery causal verification preregister

State-only full-model regression after the narrowly qualified T3 mHC dispatch fix.

Frozen from attempt034: exact prompt/oracle, temperature0/seed1, cap64, activation after8 outputs, B1/B2/B4 clean packets, B4 corrupt-first index0 and B4 corrupt-last index2, frozen full-vocabulary gates (`top1 exact`, finite, rel-L2<=0.005, max-abs<=0.125), rowwise routed and mHC controls.

Added requirement: `diagnostic-B2-corrupt-only` with K=1 / corrupt draft index0 must prove one real rejection, complete recovery and all expected common-prefix logits. Dispatch validation covers every decode packet including shortened T2/T3 tails.

`state_only=true`: no B1/B2/B4 performance measurements are executed. This attempt answers only whether the original attempt034 corrupt-first failure disappears when the missing T3 mHC path is enabled, while preserving clean/corrupt-last and explicitly qualifying B2 recovery.

No oracle, threshold, cap, sampling, model, weight, engine, cache policy, native HIP, shared expert, TP reduction, MADV_RANDOM or WO_B change. No DSpark integration.
