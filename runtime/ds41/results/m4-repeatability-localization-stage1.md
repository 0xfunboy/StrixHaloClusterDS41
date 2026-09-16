# DS41 M4 repeatability localization stage 1

Status: **FIRST_DIVERGENCE_LAYER2_OUTPUT**.

- Same-rank effective input ids/positions are bit-exact across requests; real chunks are 1023+565.
- Layers0 and1: every captured final-row field is bit-exact.
- First difference on both ranks: `layer2.hidden_states`, max_abs `0.0029296875`, rel-L2 `0.005481852104501179`. Layer2 residual/mix fields also differ after that layer.
- Final hidden rel-L2 `0.1502243`; raw target logits at logical position1587 differ (max_abs2.125, rel-L20.1293036). Top1/top2 ids remain the same in this capture.
- V2 DSpark rejection sampler is configured `raw_logprobs`; when `needs_logits_processing=false`, API logprobs are computed from the raw target logits. The observed first-token distribution change is therefore upstream of processors/reporting.
- Rank0/rank1 reported localization metrics are identical.

NEXT: capture only internal layer2 boundaries (mHC-attn, norm, attention, mHC-post, mHC-FFN, norm, FFN) for two identical M4 requests; no new broad capture or M8 work.
