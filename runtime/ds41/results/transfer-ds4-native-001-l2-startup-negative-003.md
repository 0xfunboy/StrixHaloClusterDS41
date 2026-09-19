# TRANSFER DS4 → NATIVE 001 — L2 M1 startup negative 003

**Verdict:** cache1 did not reach READY. **0 L2 quality requests were sent.** The previous `head.weight_type` fault stayed fixed and the immediate loader OOM from negative002 changed into a kernel-level SVM residency stall.

Startup identity:
- epoch `1789780947338481919`
- rank0 InvocationID `d56b8c975dce4322832e0f79e2911ef3`
- rank1 InvocationID `b3607ceea15b4650a63925859c3f5217`
- release `native-antirez-m1-transfer001-cache1`
- attempt `transfer-ds4-native-001-l2-m1-cache1`

Both ranks passed Antirez identity, native Engram runtime hash contract and entered GGUF `load_weights`. The cache1 per-tensor MADV/FADV policy kept Linux `MemAvailable` around 58 GiB and file cache approximately bounded while tens of GiB were read.

At 03:26:30–03:26:31 NODE02 emitted **543 captured kernel lines**:
`amdgpu: SVM mapping failed, exceeds resident system memory limit`.

ROCm also logged an attempted 744,488,960-byte allocation with almost no allocator-reported free space. Rank1 then remained in uninterruptible `D` state, observed in `lock_mm_and_find_vma` / `folio_wait_bit_common`, with CPU time and read progress effectively stalled for several minutes. The captured state still showed ~58.27 GiB Linux MemAvailable and GTT 67,243,618,304 / 128,849,018,880 bytes, so this is not a plain exhaustion of Linux reclaimable RAM.

Local driver state has `amdgpu.no_system_mem_limit=N`; no driver/module/BIOS setting is changed by this campaign.

**Interpretation:** file-cache eviction alone is insufficient because the tensor exposed to the GPU copy path is still backed by the GGUF mmap, so AMD HMM/SVM residency is engaged. The next bounded L2a correction is to stage each ordinary target tensor into **anonymous CPU memory** before the GPU consumer sees it, then discard the source mmap pages. Largest ordinary target tensor is only ~1.384 GiB; the ~94.4 GiB native Engram tables remain outside this iterator. No weights, quantization, driver, swap, global cache or GGUF bytes change.

Recovery:
- whole pair stopped through lifecycle: `DS41_OFF_VERIFIED`
- qualified DS4 DOCUMENT PROFILE 002 restored **READY / HTTP200**
- K2 remains OFF
- DS4 restore InvocationIDs: coordinator `92e5b2744c5d49e993ef596eb7dcbc61`, worker `131e990401c54bb792f706732495d336`

Raw evidence is preserved under `runtime/ds41/transfer-ds4-native-001/l2/cache1-*`.
