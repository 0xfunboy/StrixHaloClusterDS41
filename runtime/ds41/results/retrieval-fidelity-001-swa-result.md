SWA recent-KV: PASS; packet verificati: code2k1588 request0 rank0+rank1 chunk0+chunk1; prima discordanza: nessuna entro il gate.

# RETRIEVAL FIDELITY 001 — SWA recent-KV result

- Association proof PASS on both ranks: chunk0 logical `895..1022`, chunk1 `1460..1587`; both map exactly to the corresponding `kv_current_chunk` tail rows.
- 512 SWA rows checked total. NoPE448: **0 tokens outside the frozen upstream gate**, worst token ratio to bound `1.000000`, diagnostic per-block outliers `0`; max absolute element error `0.0625`.
- RoPE64: **0 tokens outside gate**, worst `1 BF16 ULP`.
- Same frozen code2k1588 prompt: `DISTANT_FACT_END=5` falls around token1490..1498 and is therefore inside the verified final SWA range1460..1587. This coordinate is from code2k1588 itself, not transferred from discriminator1571.
- Decision: `SWA_RECENT_KV_CONFORMS_AT_SAVED_CODE2K1588_LAYER2_PACKETS`. This is component-only: no claim about other layers, compressed/indexer context, discriminator1571, or overall model quality.
- Service was not changed by this gate. No inference, GPU replay, model load, or runtime build occurred.
