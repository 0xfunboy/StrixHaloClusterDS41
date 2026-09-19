"""DS41 V4.1 attention-parity compatibility helpers.

Opt-in only via DS41_V41_ATTN_PARITY=1.  This module never changes the default
V4.1 path.  It preserves V4.1 QAT semantics while allowing gfx1151 to keep
already-QDQ values in BF16 attention caches and to consume MXFP4 indexer Q/K
with a model-level ROCm compatibility scorer.
"""
from __future__ import annotations

import os
import torch


def enabled() -> bool:
    return os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1"


def window_fp8_block32_qdq(x: torch.Tensor) -> torch.Tensor:
    """R1c/R2 full-vector FP8 block32 UE8M0 QDQ, returned as BF16."""
    if x.dtype != torch.bfloat16 or x.shape[-1] != 512:
        raise RuntimeError(f"window parity expects bf16 [...,512], got {x.dtype} {tuple(x.shape)}")
    xf = x.to(torch.float32).reshape(-1, 16, 32)
    amax = xf.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    q = torch.clamp(xf / scale.unsqueeze(-1), -448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.to(torch.float32) * scale.unsqueeze(-1)).to(torch.bfloat16).reshape_as(x)


def compressed_nvfp4_block16_qdq(x: torch.Tensor) -> torch.Tensor:
    """NVFP4/E2M1 block16 with E4M3 scales; store only the QDQ BF16 values."""
    if x.dtype != torch.bfloat16 or x.shape[-1] != 512:
        raise RuntimeError(f"compressed parity expects bf16 [...,512], got {x.dtype} {tuple(x.shape)}")
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        ref_nvfp4_quant_dequant,
    )
    flat = x.reshape(-1, 512)
    global_scale = torch.ones((1,), dtype=torch.float32, device=x.device)
    out = ref_nvfp4_quant_dequant(flat, global_scale, 16)
    return out.to(torch.bfloat16).reshape_as(x)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_mxfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 + group32 UE8M0 to BF16 without CPU data movement."""
    if values.dtype != torch.uint8:
        raise TypeError(f"MXFP4 values must be uint8, got {values.dtype}")
    low = values & 0x0F
    high = (values >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    table = torch.tensor(
        _E2M1 + tuple(-v for v in _E2M1),
        dtype=torch.float32,
        device=values.device,
    )
    decoded = table[codes.to(torch.long)]
    if scales.dtype == torch.int32:
        scale_bytes = scales.contiguous().view(torch.uint8)
        scale_bytes = scale_bytes.reshape(*scales.shape, 4)
    elif scales.dtype == torch.uint8:
        scale_bytes = scales
    else:
        raise TypeError(f"MXFP4 scales must be int32/uint8, got {scales.dtype}")
    scale = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
    scale = torch.repeat_interleave(scale, 32, dim=-1)
    if scale.shape != decoded.shape:
        raise RuntimeError(f"MXFP4 geometry mismatch values={decoded.shape} scale={scale.shape}")
    return (decoded * scale).to(torch.bfloat16)


def _valid_index_copy(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    if cache.dtype != torch.bfloat16 or cache.shape[-1] != values.shape[-1]:
        raise RuntimeError("attention-parity BF16 cache geometry mismatch")
    flat = cache.reshape(-1, cache.shape[-1])
    valid = slots >= 0
    selected_slots = slots[valid].to(torch.long)
    selected_values = values[valid]
    if selected_values.numel() != 0:
        flat.index_copy_(0, selected_slots, selected_values)


def store_window_qdq(cache: torch.Tensor, slots: torch.Tensor, post_rope_kv: torch.Tensor) -> None:
    _valid_index_copy(cache, slots, window_fp8_block32_qdq(post_rope_kv))


def store_compressed_qdq(
    cache: torch.Tensor, slots: torch.Tensor, post_rope_latent: torch.Tensor
) -> None:
    _valid_index_copy(cache, slots, compressed_nvfp4_block16_qdq(post_rope_latent))


def qdq_compressed_cache_inplace(
    cache: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
) -> None:
    """QDQ only rows the plain BF16 writer just published at group boundaries."""
    if cache.dtype != torch.bfloat16 or cache.shape[-1] != 512:
        raise RuntimeError("compressed parity requires BF16 512-wide cache")
    valid = (slots >= 0) & (((positions[: slots.numel()] + 1) % compress_ratio) == 0)
    selected_slots = slots[valid].to(torch.long)
    flat = cache.reshape(-1, 512)
    rows = flat.index_select(0, selected_slots)
    if rows.numel() == 0:
        return
    flat.index_copy_(0, selected_slots, compressed_nvfp4_block16_qdq(rows))


def gather_bf16_paged(
    out: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Vectorized GPU gather from a BF16 paged cache into the existing workspace."""
    if cache.dtype != torch.bfloat16 or cache.shape[-1] != out.shape[-1]:
        raise RuntimeError("BF16 parity gather cache/output geometry mismatch")
    cap = out.shape[1] - offset
    if cap <= 0:
        return
    j = torch.arange(cap, device=cache.device, dtype=torch.long)[None, :]
    lens = seq_lens.to(torch.long)
    if gather_lens is None:
        take = lens
        start = torch.zeros_like(lens)
    else:
        take = gather_lens.to(torch.long)
        start = lens - take
    pos = start[:, None] + j
    valid = j < take[:, None]
    safe_pos = torch.where(valid, pos, torch.zeros_like(pos))
    block_col = safe_pos // block_size
    block_col = block_col.clamp_max(block_table.shape[1] - 1)
    physical_block = torch.gather(block_table.to(torch.long), 1, block_col)
    slot = physical_block * block_size + safe_pos % block_size
    rows = cache.reshape(-1, cache.shape[-1]).index_select(0, slot.reshape(-1))
    rows = rows.reshape(slot.shape[0], slot.shape[1], cache.shape[-1])
    rows = torch.where(valid[..., None], rows, torch.zeros((), dtype=rows.dtype, device=rows.device))
    out[:, offset : offset + cap].copy_(rows)


def gather_mxfp4_indexer_k(
    cache: torch.Tensor, slots: torch.Tensor, head_dim: int = 128
) -> torch.Tensor:
    """Gather row-major/segregrated MXFP4 K rows and dequantize to BF16."""
    if cache.dtype != torch.uint8 or cache.ndim != 3:
        raise RuntimeError(f"MXFP4 index cache must be uint8 3-D, got {cache.dtype} {cache.shape}")
    if head_dim != 128:
        raise RuntimeError(f"attention parity currently freezes index head_dim=128, got {head_dim}")
    block_size = cache.shape[1]
    value_bytes = head_dim // 2
    scale_bytes = head_dim // 32
    safe = slots.clamp_min(0).to(torch.long)
    block = safe // block_size
    pos = safe % block_size
    raw = cache.reshape(cache.shape[0], -1)
    voff = pos[..., None] * value_bytes + torch.arange(
        value_bytes, device=cache.device, dtype=torch.long
    )
    soff = block_size * value_bytes + pos[..., None] * scale_bytes + torch.arange(
        scale_bytes, device=cache.device, dtype=torch.long
    )
    v = raw[block[..., None], voff]
    s = raw[block[..., None], soff]
    out = dequant_mxfp4(v, s)
    return torch.where((slots >= 0)[..., None], out, torch.zeros((), dtype=out.dtype, device=out.device))


def _prefill_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    # Compatibility scorer: arithmetic happens on GPU; no token data leaves device.
    # Accumulate one index head at a time: same scalar formula, but the peak
    # scratch is MxN instead of MxHxN.  This is a qualification path, so we
    # deliberately trade launch count for a bounded memory contract.
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    wf = weights.to(torch.float32)
    logits = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float32, device=q.device)
    kt = k.transpose(0, 1).to(torch.float32)
    for h in range(q.shape[1]):
        logits.add_(
            torch.relu(torch.matmul(q[:, h].to(torch.float32), kt))
            * wf[:, h : h + 1]
        )
    n = torch.arange(k.shape[0], device=k.device, dtype=starts.dtype)[None, :]
    valid = (n >= starts[:, None]) & (n < ends[:, None])
    return logits.masked_fill(~valid, -torch.inf)


def _decode_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    # q=[B,N,H,D], k=[B,K,D]
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    wf = weights.to(torch.float32)
    logits = torch.zeros(
        (q.shape[0], q.shape[1], k.shape[1]), dtype=torch.float32, device=q.device
    )
    kt = k.transpose(1, 2).to(torch.float32)
    for h in range(q.shape[2]):
        dots = torch.matmul(q[:, :, h].to(torch.float32), kt)
        logits.add_(torch.relu(dots) * wf[:, :, h : h + 1])
    lens = seq_lens
    if lens.ndim == 1:
        lens = lens[:, None]
    if lens.shape[1] == 1 and q.shape[1] != 1:
        lens = lens.expand(-1, q.shape[1])
    col = torch.arange(k.shape[1], device=k.device)[None, None, :]
    return logits.masked_fill(col >= lens.to(torch.long)[..., None], -torch.inf)


def _gather_index_cache_dense(
    cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor, max_len: int
) -> torch.Tensor:
    b = block_table.shape[0]
    pos = torch.arange(max_len, device=cache.device, dtype=torch.long)[None, :].expand(b, -1)
    block_size = cache.shape[1]
    block_col = (pos // block_size).clamp_max(block_table.shape[1] - 1)
    physical = torch.gather(block_table.to(torch.long), 1, block_col)
    slots = physical * block_size + pos % block_size
    k = gather_mxfp4_indexer_k(cache, slots)
    valid = pos < seq_lens.reshape(b, -1)[:, -1:].to(torch.long)
    return torch.where(valid[..., None], k, torch.zeros((), dtype=k.dtype, device=k.device))


def rocm_mxfp4_indexer(op, hidden_states, q_quant, k, weights):
    """ROCm compatibility consumer for the existing V4.1 MXFP4 Q/K producers."""
    if not enabled():
        raise RuntimeError("attention-parity indexer called while opt-in is disabled")
    if not isinstance(q_quant, tuple) or len(q_quant) != 2:
        raise RuntimeError("attention parity requires packed MXFP4 indexer Q")
    if op.dcp_world_size != 1:
        raise RuntimeError("attention parity 002 does not qualify DCP")
    from vllm import _custom_ops as ops
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _apply_candidate_mask,
        _select_candidate_blocks,
    )
    from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

    q_values, q_scale = q_quant
    q_bf16 = dequant_mxfp4(q_values, q_scale)
    ctx = get_forward_context()
    metadata = ctx.attn_metadata
    if not isinstance(metadata, dict):
        return op.topk_indices_buffer
    meta = metadata[op.k_cache.prefix]
    outbuf = op.topk_indices_buffer
    outbuf[: hidden_states.shape[0]] = -1

    if meta.num_prefills > 0:
        pm = meta.prefill
        if pm is None:
            raise RuntimeError("missing prefill metadata")
        for chunk in pm.chunks:
            total = int(chunk.total_seq_lens)
            token_to_seq = chunk.token_to_seq.to(torch.long)
            ar = torch.arange(total, device=op.k_cache.kv_cache.device, dtype=torch.long)
            starts = chunk.cu_seq_lens.to(torch.long)[token_to_seq]
            local = ar - starts
            bcol = (local // op.k_cache.kv_cache.shape[1]).clamp_max(
                chunk.block_table.shape[1] - 1
            )
            pblock = chunk.block_table.to(torch.long)[token_to_seq, bcol]
            slots = pblock * op.k_cache.kv_cache.shape[1] + local % op.k_cache.kv_cache.shape[1]
            keys = gather_mxfp4_indexer_k(op.k_cache.kv_cache, slots)
            logits = _prefill_mqa_logits(
                q_bf16[chunk.token_start : chunk.token_end],
                keys,
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks.to(torch.long),
                chunk.cu_seqlen_ke.to(torch.long),
            )
            if op.candidate_blocks is not None:
                cand = op.candidate_blocks[chunk.token_start : chunk.token_end]
                if op.candidate_write:
                    _select_candidate_blocks(
                        logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                        cand.shape[1], op.candidate_block_size, cand,
                    )
                else:
                    _apply_candidate_mask(
                        logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                        cand, op.candidate_block_size,
                    )
            top = outbuf[chunk.token_start : chunk.token_end, : op.topk_tokens]
            torch.ops._C.top_k_per_row_prefill(
                logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, top,
                logits.shape[0], logits.stride(0), logits.stride(1), op.topk_tokens,
            )

    if meta.num_decodes > 0:
        dm = meta.decode
        if dm is None:
            raise RuntimeError("missing decode metadata")
        decode_lens = dm.decode_lens
        n_decode = meta.num_decode_tokens
        if dm.requires_padding:
            qpad = pack_seq_triton(q_bf16[:n_decode], decode_lens, pad_value=0)
        else:
            qpad = q_bf16[:n_decode].reshape(decode_lens.shape[0], -1, *q_bf16.shape[1:])
        batch, next_n = qpad.shape[:2]
        num_rows = batch * next_n
        max_len = op.max_model_len
        keys = _gather_index_cache_dense(
            op.k_cache.kv_cache, dm.block_table, dm.seq_lens, max_len
        )
        w = weights[:num_rows].reshape(batch, next_n, -1)
        logits = _decode_mqa_logits(qpad, keys, w, dm.seq_lens).reshape(num_rows, max_len)
        if op.candidate_blocks is not None:
            visible = dm.seq_lens.reshape(-1)
            if visible.numel() != num_rows:
                visible = visible.repeat_interleave(next_n)
            visible = visible[:num_rows].to(torch.int64)
            starts = torch.zeros_like(visible)
            cand = op.candidate_blocks[:num_rows]
            if op.candidate_write:
                _select_candidate_blocks(
                    logits, starts, visible, cand.shape[1],
                    op.candidate_block_size, cand,
                )
            else:
                _apply_candidate_mask(
                    logits, starts, visible, cand, op.candidate_block_size,
                )
        top = outbuf[:num_rows, : op.topk_tokens]
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, dm.seq_lens, top, num_rows,
            logits.stride(0), logits.stride(1), op.topk_tokens,
        )
        if dm.requires_padding:
            top_unpacked = unpack_seq_triton(
                top.reshape(batch, next_n, top.shape[-1]), decode_lens
            )
            outbuf[:n_decode, : top_unpacked.shape[-1]] = top_unpacked

    return outbuf


def bf16_sparse_decode(
    *,
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lens: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    """Compact selected BF16 cache rows then reuse the qualified sparse prefill math."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    b, main_w = main_indices.shape
    extra_w = 0 if extra_indices is None else extra_indices.shape[1]
    width = main_w + extra_w
    d = main_cache.shape[-1]
    compact = torch.zeros((b, width, d), dtype=torch.bfloat16, device=q.device)

    mi = main_indices.clamp_min(0).to(torch.long)
    main_rows = main_cache.reshape(-1, d).index_select(0, mi.reshape(-1)).reshape(b, main_w, d)
    main_valid = torch.arange(main_w, device=q.device)[None, :] < main_lens.to(torch.long)[:, None]
    compact[:, :main_w].copy_(torch.where(main_valid[..., None], main_rows, torch.zeros((), dtype=main_rows.dtype, device=q.device)))

    if extra_cache is not None and extra_indices is not None and extra_lens is not None:
        ei = extra_indices.clamp_min(0).to(torch.long)
        extra_rows = extra_cache.reshape(-1, d).index_select(0, ei.reshape(-1)).reshape(b, extra_w, d)
        ej = torch.arange(extra_w, device=q.device)[None, :]
        extra_valid = ej < extra_lens.to(torch.long)[:, None]
        dest = main_lens.to(torch.long)[:, None] + ej
        dest = dest.clamp_max(width - 1)
        compact.scatter_(
            1,
            dest[..., None].expand(-1, -1, d),
            torch.where(extra_valid[..., None], extra_rows, torch.zeros((), dtype=extra_rows.dtype, device=q.device)),
        )

    lens = main_lens.to(torch.int32)
    if extra_lens is not None:
        lens = lens + extra_lens.to(torch.int32)
    base = torch.arange(b, device=q.device, dtype=torch.int32)[:, None] * width
    dense_indices = base + torch.arange(width, device=q.device, dtype=torch.int32)[None, :]
    rocm_sparse_attn_prefill(
        q=q,
        kv=compact.reshape(-1, 1, d),
        indices=dense_indices,
        topk_length=lens,
        scale=scale,
        head_dim=q.shape[-1],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        attn_sink=attn_sink,
        output=output,
    )
