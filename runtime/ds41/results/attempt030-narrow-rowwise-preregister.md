# Attempt030: narrow downstream localization after routed-M>1 causal fix

Purpose: locate the **second** position42 M>1-vs-M1 divergence already implied by attempt028, without repeating attempt029's 40-layer capture.

Frozen target/reference and scheduler protocol are inherited unchanged from attempt027/029: same 35-token prompt, 64-token oracle, activation after 8 outputs, widths D1/B2/B4, temperature0, seed1, eager TP2+EP2, DenseFix + rank-local Engram2, context4096, prefix caching OFF. Final logit gates remain exact top1, finite, rel-L2<=0.005, max-abs<=0.125 and rank agreement; diagnostic tensors do not relax them.

D1 uses the promoted M1 path unchanged. B2/B4 additionally set only the existing diagnostic `DS41_NATIVE_HIP_MOE_ROWWISE=1`, whose model-free reproducer proved the layer0 routed shape-effect is removed on identical input/routing. Shared expert and the normal batched/final TP reduction remain in their original locations; no per-row collective or local routing renormalization.

Capture is restricted to model position42 and only:
- layer0: `ffn_out`;
- layer1 entry state: `state_x`, `state_pre_mix`, `state_post_mix`, `state_res_mix`, `state_residual`;
- layer1 Engram output: `engram_out`;
- layer1 forward boundaries: `layer_entry`, `attn_norm_in/out`, `attn_in/out`, `ffn_norm_in/out`, `ffn_in/out`.

This permits the next interval to be distinguished without another broad dump: if entry state and Engram output are exact but `attn_norm_in` differs, localize the second cause to layer1 delayed-mHC pre; otherwise follow the actual first differing captured field. Capture copies make attempt030 timing diagnostic/N/A. Reject/partial/rollback and block-cost timing remain forbidden until full fidelity passes.

No new weights, DSpark, engine, dependency, driver, network, GLM or UI work. Pair-safe supervisor and cleanup are mandatory; GLM remains OFF and gateway available.
