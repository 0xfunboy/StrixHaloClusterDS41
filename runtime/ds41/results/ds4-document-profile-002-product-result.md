# DS4 DOCUMENT PROFILE 002 — product result

Status: **PASS after source-contract normalization of one runner-only false negative**.

The raw product runner is preserved as FAIL because its `off_no_autoload` assertion expected HTTP409. The deployed gateway source intentionally returns **HTTP503 / code=model_not_ready** whenever lifecycle is not READY. During the gate the lifecycle and controller were both OFF and the chat did not autoload DS4, so the actual mandate requirement — reject chat while OFF without autoload — passed.

Other product gates passed:
- unauthorized chat 401;
- authenticated lifecycle READY;
- real code2k and docs2k document requests through protected gateway + corrected tokenizer, both exact PASS;
- non-stream LOW exact response;
- SSE reasoning/final separation exact response;
- cancel/drain followed by DRAIN-OK;
- whole-pair OFF and ON, followed by ON-OK.

The apparent max_tokens anomaly was false attribution. The two simple max128 requests each ended naturally after 20 completion tokens. The 2048-token backend generation was the explicit cancel/drain request, whose configured cap was 2048.

**Service limit:** client cancellation detaches the SSE client but the gateway intentionally drains the backend before releasing admission. That drain consumed the full 2048-token request and ~135.99s before DRAIN-OK could run. This is usable fail-safe drain semantics, not backend abort semantics.

Qualified product scope remains the verified document LOW 2K class, maximum observed passing document prompt 2039 tokens. Engine ctx16384 is not a quality claim. Soak is admitted.
