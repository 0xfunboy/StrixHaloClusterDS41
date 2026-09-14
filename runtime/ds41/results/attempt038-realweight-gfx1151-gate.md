# DS41-Q2-001 attempt038 — real-weight gfx1151 draft gate

**Status:** `PASS / LOAD-TIME BF16 EXPERT FALLBACK QUALIFIED`

Official shard tensors were executed on both 8060S/gfx1151 nodes. MXFP8 `mtp.0.attn.wq_a` selected `EmulationMxfp8LinearKernel`; runtime BF16 weight and output are bit-exact to an independent E4M3×UE8M0 32x32 reference.

Native MXFP4 MoE cannot be instantiated in this environment: AITER excludes gfx1151 and the pinned Triton MXFP4 backend requires optional `triton_kernels`, which is absent. No dependency was installed. The authorized fallback dequantizes immutable MXFP4+UE8M0 weights exactly to BF16 once at load and executes the existing GPU Triton BF16 fused-experts path. Rank0 expert output: rel-L2 **0.003684307**, max-abs **0.03125**. Rank1: rel-L2 **0.003525175**, max-abs **0.03125**. Frozen gates rel-L2<=0.005/max-abs<=0.125/finite all PASS.

This gate authorizes only the experimental DSpark sidecar fallback; target GGUF math remains unchanged.
