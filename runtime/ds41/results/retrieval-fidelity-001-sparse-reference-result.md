# RETRIEVAL FIDELITY 001 — sparse-attention independent reference

**PASS / component scope only.** Saved real code2k request0, both TP ranks, both prefill chunks were replayed CPU-only against the vendored independent Torch sparse-attention formula. Gate was frozen from upstream: `atol=rtol=0.02`.

All **65,536** output elements pass; 0 outside tolerance. Worst max-abs is **0.0078125** and worst rel-L2 **0.00114327**. No model was loaded, no generation request was sent, and no GPU kernel was launched by this reference check.

Decision: `SPARSE_ATTENTION_ARITHMETIC_EXCLUDED_AT_CAPTURED_LAYER2_PACKETS`. This does **not** validate the semantic correctness of indexer/top-k selection, nor the full model response. Next work may inspect only already-saved indexer/selection evidence before considering another request.
