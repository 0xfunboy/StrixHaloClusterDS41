# TRANSFER DS4 → NATIVE 001 — L2 anon1 preflight

**PASS on NODE01 and NODE02.** No model load and no L2 quality request occurred during this preflight.

New isolated release:
`/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001-anon1`

Code provenance: `c8e93a2`. Attempt: `transfer-ds4-native-001-l2-m1-anon1`.

The Antirez-only loader now stages each ordinary target tensor into one writable C-contiguous anonymous CPU buffer **before** PyTorch/ROCm sees it, then discards the source GGUF mmap range. Default loader behavior is unchanged outside this opt-in release.

Both nodes independently PASS:
- streaming name-map / prior `weight_type` fix;
- anonymous staging has no shared source memory, is byte-exact, and source bytes remain unchanged after staged mutation;
- real 1 MiB quantized Antirez sample stages byte-exact with no source sharing;
- progressive MADV/FADV source-page discard;
- 1038 ordinary target + 8 native Engram = 1046/1046;
- native Engram hash/token-map contract and row264 decode;
- fast Antirez Q2 identity against the existing both-node SHA receipt.

Largest ordinary target/staging allocation is 1,486,356,480 bytes (1.38427734375 GiB). The ~94.4 GiB native Engram tables stay outside the ordinary target iterator.

Cross-node hashes match:
- `launch-node.sh`: `507e1782a9cf6664520e36b48979cb47c8584e735e9173ad656fb70df4e99bc4`
- `weight_utils.py`: `5c991c5055bc1615745d4c644f93ce6d6ec69e4e22cb4c0d89f488e1ed727742`
- `gguf_stream_cache.py`: `6f28c0bb5d9794ed2051f6c1f35406820d4dd13617490d2012ed436cb82a3962`

Before switch: native OFF/NONE, qualified DS4 READY HTTP200, K2 OFF.

This is the final startup retry admitted for the localized mmap/SVM contract. If anon1 cannot reach READY, close the native L2 target as memory-contract blocked and leave qualified DS4 resident rather than changing driver/BIOS/infra.
