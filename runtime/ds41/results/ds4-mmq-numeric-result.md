# DS41 DS4 MMQ numerical admission

**PASS_ADMIT_FULLMODEL_B.** Frozen scale-aware gates pass on a real holdout not used for the 5.234x timing campaign and on zero/remote/clamp stress cases.

- resident `w13` ABI vs separate gate/up banks: every intermediate/output buffer bit-exact;
- no permanent expert-bank duplicate; shared scratch cap at T=1024: 337,666,048 B/rank (~322.0 MiB);
- holdout full FP32-dequant reference rel-L2: rank0 2.162%, rank1 0.821%;
- holdout per-row rel-L2 P95: rank0 4.365%, rank1 3.071% (frozen gate 8%);
- selected-row runtime vs independent Q8_1 emulation: 0.796% / 2.711%;
- zero activation exact zero; remote-only routes exact zero; clamp64 stress vs high-precision 1.218%;
- explicit clamp/SwiGLU/router epilogue agrees at ~1e-7 rel-L2.

This admits a full-model B experiment under the explicitly different activation contract. It does **not** establish bit-equivalence with DS41 Triton or functional model quality.
