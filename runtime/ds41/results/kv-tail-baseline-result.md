# DS41 V2 tail KV metadata control

**PASS_NO_METADATA_VIOLATION_OBSERVED.** Source `58a8911bcb953a15d23a811157350e4cc99311fe`, epoch `1789532175137092259`, exact historical tail prompt SHA `9996033c...`, 1546 prompt tokens, cache0. Semantic output remains `{"end":23}` vs expected5 and stays FAIL.

Actual K2 segmentation is **1023 + 523**, not the nominal1024+522: rank0/rank1 both cover positions `0..1022` then `1023..1545`, computed/seq lengths reach1546 exactly. Six cache groups use block sizes `[32,32,32,32,128,8]`; group5 has slot mapping disabled. All enabled groups have per-chunk position→block→slot consistency on both ranks. Group2 is the only group with physical-slot reuse across chunks; its reused old positions are256..767 while the K2 SlidingWindow retention (`window128`, extra retained1) permits skipping positions below895 before chunk2. Thus no violation is observed in captured metadata. Physical block IDs are intentionally not compared across ranks.

The initial validator that demanded exactly1024+522 is preserved as a harness negative; the owner mandate explicitly allowed a different actual segmentation to be documented. This result excludes gross observed truncation/chunk/slot corruption but does **not** establish numerical attention correctness or assign the semantic failure to weights/quantization.
