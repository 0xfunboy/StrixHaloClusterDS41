# TRANSFER DS4 → NATIVE 001 — L2 cache1 preflight

**PASS.** New isolated release:
`/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001-cache1`

Release provenance is `d5a8964`. No model was loaded and no quality request was sent during this preflight.

Both NODE01 and NODE02 pass the same CPU gate: streaming-name map PASS, progressive mmap cache-drop PASS using real `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`, target accounting 1038 ordinary + 8 native Engram = 1046/1046, native hash contract PASS and row264 decode PASS. The largest ordinary target tensor is 1,486,356,480 bytes (1.38427734375 GiB); the ~94.4 GiB native Engram tables remain outside the ordinary target iterator.

Both nodes also pass the fast Antirez identity gate for the existing 365,713,686,528-byte Q2 with frozen SHA receipt `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.

Cross-node release hashes match:
- `launch-node.sh`: `de90f4955623330b8de7274e75fa5ff8ede17edec535accc63ca9ed4d9eb9e68`
- `weight_utils.py`: `a29544c271c230a07090a24849d6359bdb0bfd08ca1b9f9dc56f8c89c57ff55b`
- `gguf_stream_cache.py`: `deb7b8af94a76cffca8778d5b93d5c4956e9d7a5d0addba8dc85b18996297ef6`

Before the switch, native is OFF / owner NONE, qualified DS4 is READY HTTP200 and K2 is OFF. The controller now targets this release with attempt `transfer-ds4-native-001-l2-m1-cache1`.

Next: commit/push this controller/preflight state, update PLAN/handoff, then perform exactly one owner-controlled DS4 OFF → cache1 M1 startup. No document request before rank0/rank1/paired HTTP200 and live release identity.
