# DS41-Q2-001 attempt035 — T3 runtime fix gate

**PASS / BOTH NODES.**

The only runtime math-dispatch change is the opt-in mHC rowwise guard `(2,4) -> (2,3,4)`. Fresh overlay reproduction matches all 8 vendor files byte-for-byte. On both gfx1151 nodes, T3 wrapper ON is bit-exact to concatenated promoted M1 for K5120 and K20480 and dispatches exactly 3 TileLang M1 projection/RMS calls + 1 fused coefficient/Sinkhorn `[3,24]` call per mHC invocation. Wrapper OFF preserves the historical `tokens_not_1` fallback.

Post-fix regression: M1 unchanged; T2/T4 unchanged; T7 exact fallback. No new kernel, .so, collective, weight or production-default change.

**NEXT:** state-only full-model verification of clean B2/B4, original B4 corrupt-first/last, plus explicit B2 single-draft rejection. No performance rerun.
