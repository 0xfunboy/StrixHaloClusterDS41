# Attempt032 rowwise mHC wrapper component gate

Status: **PASS on NODE01 and NODE02**. Fix commit `ce446ef62836980b4d4a3c9a62b4374ba3394c8d`. No model/LLM/generation.

For real saved B2/B4 position42 inputs, both initial-attention K=5120 and delayed-FFN K=20480 wrapper calls reproduce the promoted M1 row bit-exact when `DS41_MHC_ROWWISE_BLOCK=1`. B2 dispatch is exactly 2 TileLang M1 projection calls + one fused C1 call over 2 rows; B4 is 4 TileLang calls + one fused C1 call over 4 rows. No projection/coeff fallback occurs in qualified T2/T4. M1 is unchanged, T7 remains exact fallback, and the branch fails closed if either promoted projection/RMS or coefficient/Sinkhorn is disabled.

The patch overlay fresh-applies from pinned vLLM `0bfb653d...` and reproduces the complete 8-file deployed vendor delta byte-for-byte, including `disk_engram.py`. Pair remained OFF_VERIFIED/NONE-OFF.
