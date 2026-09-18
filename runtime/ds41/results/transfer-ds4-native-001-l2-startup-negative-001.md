# TRANSFER DS4 → NATIVE 001 — L2 M1 startup negative 001

Result: **FAIL_BEFORE_REQUEST_LOCALIZED_LOADER_INTEGRATION**.

The first Antirez/native-Engram target-only M1 startup acquired the pair once under epoch `1789761472122549252`. Both ranks launched with the intended release and no document request was sent.

During `load_weights`, both ranks terminated with:

`DS41 streaming loader only accepts the single language_model group, got 'head.weight_type'`

This is a localized GGUF/V4.1 streaming name-mapping defect. The GGUF iterator emits `*.weight_type` companions for packed tensors. The V4.1 outer mapper already rerooted `head.weight` and `embed.weight`, but not their quantization-type companions. The fail-closed streaming loader therefore rejected `head.weight_type` before model READY.

The negative does **not** implicate Antirez quality, Engram math, MMQ or DSpark. Before the failure, Antirez identity passed, the native Engram runtime hash contract passed, and the target accounting remained 1038 ordinary + 8 native Engram tensors = 1046/1046.

Recovery followed the owner contract: both failed DS41 units/cgroups were verified OFF, the stale DS41 receipt was reconciled through `pair.sh reconcile`, and qualified DS4 DOCUMENT PROFILE 002 was restored READY HTTP200. K2 remains OFF.

The scoped fix adds only:
- `head.weight_type -> language_model.lm_head.weight_type`
- `embed.weight_type -> ...embed_tokens.weight_type`

Both NODE01 and NODE02 CPU gates now PASS all 8 streaming name-map cases, full 1046 tensor accounting and the existing real-row native Engram decode checks. Because zero model requests were sent and the blocker is localized with a model-free regression proof, one retry of the same L2 M1 configuration is admitted; it is not an additional configuration.

Raw:
- rank0: `/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001/reports/DS41-Q2-001/transfer-ds4-native-001-l2-m1/rank0.log`
- rank1: corresponding `rank1.log`
- fix gates: `runtime/ds41/transfer-ds4-native-001/l2/cpu-gate-node{01,02}-fix1.json`
