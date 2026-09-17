# RETRIEVAL FIDELITY 001 — preregister

Single full-model discriminator: same `k2-prefill-5bdfed6` source/release and frozen extraction prompt, but target-only M1 (`speculative_config=None`). Input audit already confirms `end=5` is physically present near the final instruction and the discriminator has no median ambiguity.

Decision: M1 PASS implicates K2/spec-verifier; same FAIL excludes K2 as a necessary cause and narrows the problem to target model/runtime/weights/reference contract. No repeat or alternate prompt before this result.
