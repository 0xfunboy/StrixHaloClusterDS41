# DS4 DOCUMENT PROFILE 002 — offline input contract

Status: **PASS / model input from USABLE 001 was not altered by the sidecar bug**.

The observed CLI/sidecar +6 comes from one exact token sequence:
`[128799,3476,477,260,11502,22896]`
which is `<｜System｜>You are a helpful assistant`.

The pinned CLI one-shot supplies that default system. The OpenAI Chat server renderer uses the messages actually supplied and does **not** inject it for user-only requests. USABLE 001 therefore already sent the correct user-only model input; fixing the CPU sidecar cannot retroactively repair those semantic FAILs.

The corrected sidecar follows the server DeepSeek V4.1 renderer and matches the embedded Antirez Q2 tokenizer IDs exactly for user-only, explicit-system and multi-turn fixtures under NONE and LOW, plus the actual code/docs v2 and both new holdouts.

LOW contract:
- parser: `DS4_THINK_LOW`
- thinking enabled: true
- numeric 1–100 level: **none** (`ds4_think_mode_level=-1`)
- extra Reasoning Effort system text: none
- compared with NONE, the document text/tokens are unchanged and the generation prefix changes from final token `128822 </think>` to `128821 <think>`.

Thus DOCUMENT PROFILE 002 selects LOW because there is no demonstrated live-input defect to isolate under NONE.
