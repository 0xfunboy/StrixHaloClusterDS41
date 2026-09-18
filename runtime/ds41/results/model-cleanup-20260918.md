# Model cleanup 2026-09-18

Owner-authorized cleanup completed on NODE01 and NODE02. Removed the four explicitly named Qwen checkpoints plus the historical broken DeepSeek V4.1 MixedQ2 original. Freed **286.121 GiB per node**. DenseFix, Engram TP2, DSpark sidecar and current K2 serving remain intact; K2 verified READY after deletion.

The original MixedQ2 was safe to remove: the PLAN/source audit proved systematic corruption in FP8-derived dense BF16 tensors, while `artifact_identity.py` uses `original_model_dir` only as a path inequality guard and does not require the original payload to exist.
