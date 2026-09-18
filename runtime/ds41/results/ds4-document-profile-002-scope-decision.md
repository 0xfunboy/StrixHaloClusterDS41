# DS4 DOCUMENT PROFILE 002 — scope decision

The six-request LOW document quality gate is qualified. Context escalation stops at the first larger case:

- code4k: **INCOMPLETE_NO_FINAL**, 3486 prompt + 512 completion tokens, all 512 consumed before a final answer; prefill30.284s /115.11 tok/s, decode15.89, wall62.550s.
- code8k/code16k: not sent.

Therefore document quality is limited to the verified 2K-class perimeter, with the largest passing observed prompt at **2039 tokens**. The engine remains allocated at ctx16384; that is not a quality claim.

Per the mandate, this larger-context failure limits scope but does not revoke the smaller 6/6 document qualification. Product gateway checks are therefore admitted for the smaller verified perimeter. The raw continuation boolean `product_admitted=false` is preserved as an over-restrictive runner decision, not rewritten.
