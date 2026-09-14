# Attempt029: position42 decoder-boundary analysis

Status: **LOCALIZED_TO_LAYER0_FFN_INTERVAL**. This is CPU-only analysis of the saved attempt029 captures; no model load was repeated.

## Provenance and capture validity

- Capture source: `47519fdfb3f89012d479af782159a7300e852340`; hardened comparator: `c1ef1a075cdb1327b85873739d756e873810f30f`.
- Attempt029 rowwise-routed control was OFF. D1 step42 is `[position42]/[token126513]`; B2 is `[42,43]/[126513,297]`; B4 is `[42,43,44,45]/[126513,297,63696,271]`. Thus row0 is the same real position/input in all arms.
- Each arm has exactly 360 unique boundaries = 40 layers x 9 ordered stages, with no duplicates, missing stages or token-axis mismatches. Shapes are `[1,...]`, `[2,...]`, `[4,...]` for D1/B2/B4 respectively; row0 is therefore position42, not an assumed generic row.
- Hook snapshots are independent CPU copies (`detach().cpu().clone()`), not live views.
- Rank0 and rank1 independently produce identical reported metrics for every B2/B4 boundary. Decoder hidden-state boundaries are replicated quantities after the relevant TP reductions; no sharded weight tensor is compared across ranks.
- Timing is N/A because capture copies alter request cost.

## First localization

**Last exact boundary: layer0 `ffn_in`. First divergent boundary: layer0 `ffn_out`.**

For both B2 and B4, and independently on both ranks:

| layer | boundary | exact | differing elements | reference norm | rel-L2 | max-abs | non-finite |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | layer_entry | yes | 0/5120 | 10.02354336 | 0 | 0 | 0 |
| 0 | attn_norm_in | yes | 0/5120 | 10.02354336 | 0 | 0 | 0 |
| 0 | attn_norm_out | yes | 0/5120 | 1.43949020 | 0 | 0 | 0 |
| 0 | attn_in | yes | 0/5120 | 1.43949020 | 0 | 0 | 0 |
| 0 | attn_out | yes | 0/5120 | 54.06748199 | 0 | 0 | 0 |
| 0 | ffn_norm_in | yes | 0/5120 | 19.52683640 | 0 | 0 | 0 |
| 0 | ffn_norm_out | yes | 0/5120 | 8.99687576 | 0 | 0 | 0 |
| 0 | **ffn_in** | **yes** | **0/5120** | **8.99687576** | **0** | **0** | **0** |
| 0 | **ffn_out** | **no** | **3766/5120** | **8.76624012** | **0.006821454** | **0.0078125** | **0** |

The first numerical difference is also the first widespread/material difference: every captured boundary before it is bit-exact, while 3766/5120 (~73.55%) BF16 elements change at `ffn_out`. This statement does **not** reuse the frozen logit tolerance as an intermediate-tensor gate.

The error then propagates. At layer1 `ffn_out`, rel-L2 is `0.01040790`. Later peaks are diagnostic amplification, not new causes: B2 peaks at layer34 `ffn_out` rel-L2 `0.29450214` / max-abs `0.451171875`; B4 peaks at layer25 `ffn_out` rel-L2 `0.34013614` / max-abs `0.552734375`.

B2 and B4 row0 themselves remain equal through layer2; their first row0-to-row0 difference appears only later (layer3 `attn_norm_in`, one BF16 element). The common M>1-vs-M1 deviation therefore precedes the width-specific B2-vs-B4 split.

## Executed code interval

The localized interval is the call at `deepseek_v4_1/amd/model.py:350`, `x = self.ffn(x, input_ids)`, implemented by `DeepseekV4MoE.forward`. Within that interval the executed runner can perform only:

1. shared-expert execution on the FFN input;
2. gate linear projection;
3. router `select_experts` (top6 IDs/global weights);
4. routed expert computation;
5. shared/routed output combination;
6. the normal single TP reduction/finalization.

The current dump does **not** observe gate logits, top-k IDs/weights, shared output, routed output, or pre-reduction combined output. Therefore it localizes a code interval, not yet one single operation. Attention and the preceding mHC/FFN normalization are excluded as the first cause because their output/FFN input row is exact.

## Attempt028 interpretation corrected

Attempt028 proves only that replacing the routed expert **computation** with rowwise native-M1 calls is not sufficient to restore B2/B4 fidelity. Since layer0 `ffn_in` is now proven exact, the remaining first-layer candidates are shape-sensitive routing, shared-expert math, combine/reduction, and any routed contribution not corrected by using batched routing metadata. It is not valid yet to claim one of those individually.

## NEXT

Do not repeat the 40-layer capture. Reuse the exact saved layer0 `ffn_in` tensors to build a component reproducer for the localized FFN interval. First compare M1 vs M2/M4 row0 for gate/router and shared expert using the actual layer0 weights; compare operators on identical inputs. Reuse the existing native/Triton routed component machinery for the actual selected routes. If this reproduces the captured `ffn_out` delta, correct only the demonstrated shape-dependent operation. If an indispensable internal value still cannot be reconstructed independently, the only justified new model capture is layer0-FFN-internal at position42 (gate/topk/shared/routed/pre-reduce), not another 40-layer dump.

Full ordered table: `runtime/ds41/results/attempt029-boundary-analysis.tsv`. Local raw comparator receipts remain under `reports/DS41-Q2-001/attempt029/`.
