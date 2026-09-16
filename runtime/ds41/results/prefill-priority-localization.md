# DS41 prefill priority localization

The original `code-2k-s1` remains a semantic FAIL. This work does not retry it for a passing score.

Observed existing evidence: 1588 prompt/computed, cached0, pair prefill147426ms, client first-final147.770s. Gateway/tokenization was ~0.30s; cost is inside engine prefill.

Targeted guard discriminator on the same resident release: exact 64-token prompt prefill5017.646ms; exact65-token prompt6499.533ms. The source changes branch exactly at `x.shape[0] <=64` for `DS41_EP_SKIP_REMOTE`.

Real layer0 MixedQ2 component A/B, full4 reference vs EP2 sum:
- M65: current critical47.777ms -> candidate29.098ms (-39.10%), candidate max_abs=0.
- M1024: current critical662.134ms -> candidate336.553ms (-49.17%), candidate max_abs=0.

The change is only the existing EP skip-remote MMVQ guard `<=64 -> <=1024`, matching the frozen scheduler max prefill chunk. No new kernel/model/precision/driver/network. Decision: admit exactly one real code2K before/after comparison after an isolated release switch. Separately preregister an extraction-only correctness discriminator; speed and semantic correctness remain distinct.
