# Attempt033: combined routed+mHC B2/B4 fidelity verification

Status: **PREREGISTERED / READY AFTER PEER PREFLIGHT**.

Candidate math commit: `ce446ef62836980b4d4a3c9a62b4374ba3394c8d`. D1 is the unchanged promoted M1 target. B2/B4 enable both opt-in fidelity controls only during packet requests: (1) routed native M1 per row, preserving shared expert and one batched TP reduction; (2) mHC P1 projection/RMS per row plus one batched C1 coefficient/Sinkhorn call. No DSpark/drafter is loaded.

Reuse byte-identical attempt031/027 speed prompt, 64-token oracle, activation after8 outputs, temperature0, seed1, context4096, prefix cache OFF and final gates: finite logits, exact top1, per-position rel-L2<=0.005, max-abs<=0.125, exact oracle outputs and rank agreement. Keep the attempt031 narrow capture only: layer0 `ffn_out`; layer1 state_x/pre/post/res-mix/residual, layer_entry, engram_out, attn norm/input/output and FFN norm/input/output. Tensor-copy requests are diagnostic and never performance samples.

Sequence remains D1 warmup+diagnostic, B2 warmup+diagnostic, B4 warmup+diagnostic. If B4 clean fidelity passes, execute the frozen corrupt-first/corrupt-last reject controls, then only qualifying widths may enter the existing 3x32 block measurement. If fidelity fails, no block timing/DSpark claim; preserve the narrow boundary evidence and localize only the residual interval needed. Pair-safe supervisor and whole-pair cleanup are mandatory.
