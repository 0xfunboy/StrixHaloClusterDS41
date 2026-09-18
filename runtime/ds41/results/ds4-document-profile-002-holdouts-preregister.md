# DS4 DOCUMENT PROFILE 002 — holdout preregister

Frozen before model load.

- **Holdout A**: 5 sections, LOW prompt 1409 tokens, SHA `1644e146...`; reordered code sections plus HEAD/CENTER/TAIL facts and ten deterministic JSON fields.
- **Holdout B**: 7 sections, LOW prompt 2039 tokens, SHA `f072aafa...`; mixed code/docs sections with different positions/facts and the same multi-field contract.
- Expected values are validator-side only and are not inserted as answers into either prompt.
- Both holdouts run only after code2k-v2 and docs2k-v2 PASS under the frozen LOW profile.
