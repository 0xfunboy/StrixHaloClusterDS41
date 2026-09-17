# DS41 RECOVERY R1b — offline endpoint result

Status: **PASS**.

Delayed/single-pass layer2 mHC-pre + attn_norm: PASS on both ranks/chunks.

Output projection endpoint:
- chunk0 pos1022: rank0 rel-L2=0.000743023 max-abs=0.0078125; rank1 rel-L2=0.000743023 max-abs=0.0078125; targets exact=True
- chunk1 pos1587: rank0 rel-L2=0.00011714 max-abs=0.00195312; rank1 rel-L2=0.00011714 max-abs=0.00195312; targets exact=True

Decision: `R1B_ENDPOINTS_CONFORM`. This closes only the saved layer2 endpoint intervals; it is not a semantic/full-model PASS.
