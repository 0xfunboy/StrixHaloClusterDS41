# Attempt029: layer0 FFN causal reproducer

Status: **FIRST_CAUSAL_PATH_LOCALIZED_ROUTED_MGT1**. No LLM was constructed and no generation request was issued. The test uses the saved real position42 `ffn_in` tensors and real layer0 DenseFix/GGUF weights. It was repeated on both gfx1151 nodes with identical metrics.

## What is fixed by the evidence

The saved boundary analysis proved that D1/M1 and B2/B4 are bit-exact through layer0 `ffn_in`, then diverge at `ffn_out`. The model-free component reproducer separates the operations inside that interval on the same row0 input.

For both B2 and B4, on both nodes:

| component, row0 M>1 vs M1 | result |
|---|---|
| GateLinear logits | **bit-exact** |
| top-6 expert IDs | **exact same IDs** (`80,238,373,197,156,116`) |
| top-6 routing weights | **bit-exact** |
| reconstructed shared expert output | **bit-exact** |
| routed standard M>1 output | **DIFF**: 4680/5120 values, rel-L2 `0.01817124`, max-abs `0.00390625` |
| routed rowwise-native output | **bit-exact to M1** |
| reconstructed shared + rowwise-routed total | **bit-exact to reconstructed M1 total** |

Independent gguf-py dequantization plus explicit FP32/BF16 math on the actual six routes gives rel-L2 `0.01000996` for native-M1/rowwise-native versus `0.01593358` for the standard M>1 routed path. This supports the promoted M1 numerical contract on the real input; it does not claim either low-bit kernel is an exact real-number matmul.

The reconstructed standard total has M>1-vs-M1 rel-L2 `0.00675590`; the actual captured `ffn_out` has `0.00682145`. The delta vectors have cosine `0.87868`. The remaining reconstruction error is expected from reproducing two-rank TP arithmetic locally rather than replaying RCCL and every runtime wrapper; it is not used as an exact-equivalence claim.

## Causal conclusion

The **first demonstrated causal path is the routed-expert M>1 fallback** used by the packet verifier. On identical FFN input and identical routing metadata, only that component changes; replacing it with the qualified native-M1 calculation per row removes the layer0 FFN shape-effect in the component reproducer.

This implicates the routed path **as a bundle** (native Q8_1 activation quantization, low-bit kernel/reconstruction/accumulation contract versus the Triton BF16 packet path). It does **not** prove Q8_1 alone is the cause.

Attempt028 is therefore refined, not contradicted: rowwise native M1 is sufficient to remove the **first layer0 FFN divergence**, but attempt028's full logits still fail at position42. A second downstream M>1-vs-M1 divergence remains after layer0 once the routed path is aligned.

## NEXT

Do not repeat attempt029's 40-layer dump. Keep the rowwise routed control opt-in and capture only the narrow downstream boundary under that correction, beginning with layer0 `ffn_out` and layer1 mHC/attention/FFN boundaries. If layer0 `ffn_out` is exact and layer1 `attn_norm_in` is the first difference, isolate the layer1 delayed-mHC pre/post inputs/carry and reuse the promoted M1 primitives per row. If not, follow the actual first differing boundary. No block timing/reject qualification until full fidelity passes.
