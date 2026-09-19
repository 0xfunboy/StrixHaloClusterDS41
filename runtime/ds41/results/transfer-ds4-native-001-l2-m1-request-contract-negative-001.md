# TRANSFER DS4 → NATIVE 001 — L2 M1 request-contract setup negative 001

The first quality runner terminal is preserved exactly as written, but it is **not a semantic M1 failure**.

Both attempted cases were rejected by the paired coordinator with HTTP400 `unknown model` before inference:
- code2k: 0.0065s, no usage/reasoning/final
- docs2k: 0.00038s, no usage/reasoning/final

The two rank APIs expose `DeepSeek-V4.1-Flash-Q2-Native-M1`; the generic resident paired coordinator still exposed and injected `DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2`. Therefore **accepted model requests = 0** and the frozen quality budget remains 0/6. The raw registry/terminal are not reset or rewritten.

Correction is control-plane-only:
- original K2 config left untouched;
- original `ds41-haloclu-pair.service` stopped while idle;
- transient `ds41-transfer-l2-pair.service`, InvocationID `f0019e8e735b46378e6940e639e73455`, serves 18221 with model identity `DeepSeek-V4.1-Flash-Q2-Native-M1`;
- health is ok, busy=false, both ranks true, and `/v1/models` now matches the two rank APIs.

A distinct runner namespace `l2-m1-run2` and distinct X-Request-IDs are used for the actual quality run. The old `l2-m1` terminal remains setup evidence.
