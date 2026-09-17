# Final quality discriminator comparison

- Identical frozen prompt SHA256: `f5543bba60bbf849bb106bf9efa9ca5381b88a4c4c9281cf60de37a767258745`.
- Candidate `9c13117f`: FAIL. Rollback `5bdfed6`: FAIL.
- Returned content exact across candidate and rollback: **True**.
- Both report begin=17, middle=23, end=29 and the same incomplete/wrong file list.
- Decision: **SHARED_PREEXISTING_QUALITY_FAILURE_NOT_CANDIDATE_REGRESSION**. The retrieval/context failure predates MMQ+Engram; this does not make the candidate quality-qualified.
