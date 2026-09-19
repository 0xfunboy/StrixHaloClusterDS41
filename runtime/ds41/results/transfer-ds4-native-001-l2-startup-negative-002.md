# TRANSFER DS4 → NATIVE 001 — L2 M1 startup negative 002

**Verdict:** setup/load failure before READY; **0 model requests sent**. The prior `head.weight_type` fault did not recur.

Retry epoch `1789761988271104620` used rank0 InvocationID `2d6a594ff8124a84aa029b18ef6cfbbd` and rank1 InvocationID `dd4432be158e45ddb491d355786de6a1`. Antirez identity and the native Engram runtime hash contract passed, then NODE02 failed in GGUF MoE parameter materialization with `torch.OutOfMemoryError`: the next allocation was 710 MiB, reported UMA capacity 120 GiB, free 4.28 MiB, PyTorch allocated 62.13 GiB and reserved-but-unallocated 76.10 MiB.

The localized loader defect is the residency policy for the single 365,713,686,528-byte GGUF. The existing `DS41_DROP_SHARD_CACHE=1` discards clean mmap pages only after the whole shard finishes, which is too late for this one-file artifact. The two native Engram tables are about 94.4 GiB each but are excluded from the ordinary target iterator; the largest ordinary target tensor is about 1.384 GiB. Therefore the next intervention is bounded **per-consumed-tensor clean-page eviction**, not a new quantization, split checkpoint, global cache drop or swap change.

The remaining rank0 was stopped through the whole-pair lifecycle and both DS41 units/cgroups were verified OFF. Qualified DS4 DOCUMENT PROFILE 002 has been restored **READY / HTTP200**; K2 remains OFF.

Raw failure evidence remains in:
`/home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001/reports/DS41-Q2-001/transfer-ds4-native-001-l2-m1/rank1.log`.

A same-configuration startup retry is admitted only after the progressive file-cache gate passes, because no L2 quality request has yet been sent.
