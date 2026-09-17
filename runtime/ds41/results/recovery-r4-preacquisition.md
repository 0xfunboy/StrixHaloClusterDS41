# DS41 RECOVERY R4 — pre-acquisition gate

Status: **PASS_ADMIT_SINGLE_Q2_ACQUISITION**.

- DS4 pin `7d0454b4...` builds on both gfx1151 nodes with bundled HIP 7.15.26333; help/eval/linking PASS; binaries bit-identical.
- Calibrated Q2 pinned to HF revision `dd8a266f...`: `365713686528` bytes (340.597 GiB), SHA256 `1ce6a8f8...`.
- GPU-visible pool: 125829120 KB (~120 GiB) each; cluster documentation expects ~80.6 GiB resident weights/rank with Engram disk-backed.
- NODE02 post-download unprivileged margin: 353.00 GiB. NODE01 post-mirror unprivileged margin: 32.31 GiB; filesystem free including reserved blocks: 125.42 GiB. No second full-copy staging is used by HF local-dir acquisition or planned in-place resumable mirror.
- USB4 TCP link 10.55.0.1↔10.55.0.2 reachable. Serving `5bdfed6` remains READY and untouched.

NEXT: one Internet copy on NODE02, pinned revision, persistent runner; verify full SHA before any mirror.
