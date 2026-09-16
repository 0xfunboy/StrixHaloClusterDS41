# BLOCK_M4/8 raw audit

A/B are comparable on source/vendor, frozen corpus, request settings, model/engine configuration and per-arm rank coherence. The paired coordinator semantic-compares both ranks with `reflect.DeepEqual`; both arms returned successfully and left poison empty.

The raw SSE does **not** preserve prompt or completion token IDs (`prompt_token_ids:null`, `token_ids:null`), so none are reconstructed by retokenization. The first observable emitted divergence is after four common completion tokens: completion token **5** emits `48` for M4 and `46` for M8.

The raw A/B files also do not directly preserve chunk ranges. The earlier real route capture on the same source/config observed 1023+565 for the 1588-token prompt, but that is supporting evidence rather than direct A/B raw proof.

Verdict: same-arm repeatability is still missing; do not attribute the output difference to M8 yet.
