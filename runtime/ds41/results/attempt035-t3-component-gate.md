# DS41-Q2-001 attempt035 — T3 mHC component gate

**PASS / MODEL-FREE / BOTH gfx1151 NODES.**

Using the real saved layer0 B4 fixture, T=3 is bit-exact to three promoted M1 calls for both initial-attention K=5120 and delayed-FFN K=20480. Each mHC invocation uses exactly three existing TileLang M1 projection/RMS calls and one existing fused coefficient/Sinkhorn call on `[3,24]`; there is no fallback and no collective per row. C1 `[3,24]` is itself bit-exact to concatenated C1 T1 row programs.

Existing regression remains PASS on both nodes: M1 unchanged, T2/T4 unchanged, T7 exact fallback. Runtime guard was still `(2,4)` during this qualification.

**Decision:** qualify only T=3 for the existing opt-in rowwise mHC fidelity wrapper; no new kernel/build and no broad batch-size expansion.
