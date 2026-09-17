# DS41 DS4 MMQ external-arena recovery gate

Status: **PASS_RECOVERY_GATE**.

B1 (`1789624782020877084`) produced no valid performance/model sample: the DS4 standalone ROCm pool reserve guard rejected an internal allocation, rank0 aborted and rank1 then observed the consequential distributed error. The recovery preallocates a bounded 64 MiB GPU arena through Torch and serves DS4 pool allocations from it; MMQ math and the frozen workload are unchanged.

Model-free gate on the already captured real T=1023 rank-local fixtures passed on both gfx1151 nodes. Each node executed three identical calls: 24 pool allocations total, 35,414,272 B high-water, 0 B left in use, exact repeat output, finite output, and exact EP2 local-id mapping. Candidate library SHA256: `7c19b29e8e1cfd37fe1fc6f8005ac59c64aa9f0b0cd7e0a897fb3a0313ed6756`.

Decision: admit exactly one new **B2** request ID after freezing this source/release. Do not replay B1 and do not erase its failure. Require full-model terminal raw plus DS4 exit counters before treating B2 as a valid performance sample.
