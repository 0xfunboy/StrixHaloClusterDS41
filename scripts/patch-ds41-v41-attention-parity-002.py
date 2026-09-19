#!/usr/bin/env python3
"""Patch one copied WOA1 release for DS41 V4.1 ATTENTION PARITY 002.

Fail-closed against the exact files from the effective L2 WOA1 release.
Default code paths remain unchanged unless DS41_V41_ATTN_PARITY=1.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

BASE_HASHES = {
    ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/attention.py": "d6719aa3044f2b3fcf37aea97c4cbb83146a639e3a6afcf314002211c3f42521",
    ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py": "3ffe978236978ca9247a1db144a5c9cb683d794cad7483f002b7f7f822024e64",
    ".vendor/vllm-dsv41/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py": "ac1c8cc89565fb90d7569170ce137e992662bb8be88e66bfb32ada3d0730ea15",
    ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py": "cc486feefe40871f010c38ee97a2217ca70d86b7f42cdfc5884399045d5ac385",
    ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/amd/rocm.py": "7f28e14337ad738d6a03675ea6eb1549cf19492d4cf255adb051736e76c20360",
    ".vendor/vllm-dsv41/vllm/v1/attention/backends/mla/sparse_swa.py": None,
    ".vendor/vllm-dsv41/vllm/v1/attention/backends/mla/indexer.py": None,
    ".vendor/vllm-dsv41/vllm/model_executor/layers/sparse_attn_indexer.py": None,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: patch anchor count={count}, expected 1")
    path.write_text(text.replace(old, new))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch-ds41-v41-attention-parity-002.py SOURCE_WORKTREE TARGET_RELEASE")
    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    if not (target / ".source-commit").is_file():
        raise RuntimeError(f"not a DS41 release: {target}")
    provenance = (target / ".source-commit").read_text().strip()
    if provenance != "9fe0496398aae695c9d2486ad7b234a8ee0fde80":
        raise RuntimeError(f"ATTN PARITY 002 requires WOA1 source 9fe0496, got {provenance}")

    # Freeze known hashes where already recorded.  The remaining three files are
    # validated structurally by exact replacement anchors below.
    for rel, expected in BASE_HASHES.items():
        p = target / rel
        if not p.is_file():
            raise RuntimeError(f"missing base file {p}")
        if expected is not None and sha(p) != expected:
            raise RuntimeError(f"base hash mismatch {rel}: {sha(p)} != {expected}")

    for rel in (
        "runtime/ds41/v41_attention_parity.py",
        "scripts/test-ds41-v41-attention-parity-002-cpu.py",
        "scripts/test-ds41-v41-attention-parity-002-device.py",
        "scripts/localize-ds41-v41-attention-parity-002-producers.py",
    ):
        src = source / rel
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    fused_indexer_q = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"
    replace_once(
        fused_indexer_q,
        '''@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
''',
        '''@triton.jit
def _fp32_to_e2m1_code_portable(x):
    """Portable OCP E2M1 RTNE code, valid on ROCm where PTX inline asm is invalid."""
    a = tl.abs(x)
    code = tl.where(
        a <= 0.25,
        0,
        tl.where(
            a < 0.75,
            1,
            tl.where(
                a <= 1.25,
                2,
                tl.where(
                    a < 1.75,
                    3,
                    tl.where(
                        a <= 2.5,
                        4,
                        tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)
    sign = ((x.to(tl.uint32, bitcast=True) >> 31) & 1).to(tl.uint8)
    return code | (sign << 3)


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # Two E2M1 values per byte: low nibble is x_lo, high nibble is x_hi.
    lo = _fp32_to_e2m1_code_portable(x_lo)
    hi = _fp32_to_e2m1_code_portable(x_hi)
    return lo | (hi << 4)
''',
    )

    attention = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/attention.py"
    replace_once(
        attention,
        '''    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout
''',
        '''    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        if os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1":
            return False
        return self.use_fp8_ds_mla_layout
''',
    )
    replace_once(
        attention,
        '''        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )
''',
        '''        if os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1":
            # Compatibility depot: cache contains only already-QDQ V4.1 values.
            # It is intentionally BF16 so gfx1151 consumers do not re-encode the
            # values through the historical hybrid FP8 writer.
            self.kv_cache_dtype, self.kv_cache_torch_dtype = (
                "bfloat16",
                torch.bfloat16,
            )
        else:
            self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
                self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
            )
''',
    )
    replace_once(
        attention,
        '''    quantize_and_insert_k_cache(
        kv_rope,
        k_cache,
        slot_mapping,
        block_size=block_size,
        is_ue8m0=True,
        use_fnuz=False,
    )
''',
        '''    if os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1":
        from runtime.ds41.v41_attention_parity import store_window_qdq

        store_window_qdq(k_cache, slot_mapping, kv_rope)
    else:
        quantize_and_insert_k_cache(
            kv_rope,
            k_cache,
            slot_mapping,
            block_size=block_size,
            is_ue8m0=True,
            use_fnuz=False,
        )
''',
    )
    replace_once(
        attention,
        '''        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
''',
        '''        uses_fp8_ds_mla_layout = (
            vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
            and not (
                os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1"
                and self.head_dim == 68
            )
        )
''',
    )

    replace_once(
        attention,
        '''        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
                False,
            )
            return q
''',
        '''        if cache_dtype == torch.bfloat16:
            if (
                os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1"
                and os.environ.get("DS41_DECOMPOSED_QKV_INSERT", "0") == "1"
            ):
                return _ds41_decomposed_q_rope_kv_insert(
                    q,
                    kv,
                    swa_kv_cache_3d,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    self.padded_heads,
                    block_size,
                )
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
                False,
            )
            return q
''',
    )

    indexer_backend = target / ".vendor/vllm-dsv41/vllm/v1/attention/backends/mla/indexer.py"
    replace_once(indexer_backend, "from dataclasses import dataclass\n", "from dataclasses import dataclass\nimport os\n")
    replace_once(
        indexer_backend,
        '''    use_fp4 = kv_dtype == "mxfp4"
    if use_fp4 and not current_platform.is_device_capability_family(100):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
    return use_fp4
''',
        '''    if os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1":
        if not current_platform.is_rocm():
            raise ValueError("DS41 attention parity MXFP4 override is ROCm-only")
        return True
    use_fp4 = kv_dtype == "mxfp4"
    if use_fp4 and not current_platform.is_device_capability_family(100):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
    return use_fp4
''',
    )

    index_store = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py"
    replace_once(index_store, "import torch\n", "import os\nimport torch\n")
    replace_once(
        index_store,
        '''    shuffle = current_platform.is_rocm() and block_size > 1
    if shuffle:
        if use_fp4_cache:
            raise NotImplementedError(
                "MXFP4 indexer K cache has no tiled ROCm layout; "
                "the ROCm readers only implement the FP8 one."
            )
''',
        '''    parity_fp4 = (
        use_fp4_cache
        and os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1"
        and current_platform.is_rocm()
    )
    # Parity 002 has its own ROCm MXFP4 consumer and deliberately keeps the
    # canonical row-major packed layout.  The historical FP8 ROCm path retains
    # the 16x16 tiled layout unchanged.
    shuffle = current_platform.is_rocm() and block_size > 1 and not parity_fp4
    if shuffle:
        if use_fp4_cache:
            raise NotImplementedError(
                "MXFP4 indexer K cache has no historical tiled ROCm layout."
            )
''',
    )

    sparse_indexer = target / ".vendor/vllm-dsv41/vllm/model_executor/layers/sparse_attn_indexer.py"
    replace_once(
        sparse_indexer,
        '''        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
''',
        '''        if self.use_fp4_cache:
            from runtime.ds41.v41_attention_parity import (
                enabled as ds41_v41_attention_parity_enabled,
                rocm_mxfp4_indexer,
            )

            if not ds41_v41_attention_parity_enabled():
                raise AssertionError("AMD MXFP4 indexer requires DS41 attention parity opt-in")
            return rocm_mxfp4_indexer(self, hidden_states, q_quant, k, weights)
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
''',
    )

    compressor = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py"
    replace_once(compressor, "import torch\n", "import os\nimport torch\n")
    replace_once(
        compressor,
        '''    _rope_plain_insert_kernel[(num_tokens,)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        fp8_scale if store_fp8 else None,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        STORE_FP8=store_fp8,
        num_warps=4,
        **launch_kwargs,
    )
''',
        '''    _rope_plain_insert_kernel[(num_tokens,)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        fp8_scale if store_fp8 else None,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        STORE_FP8=store_fp8,
        num_warps=4,
        **launch_kwargs,
    )
    if (
        os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1"
        and kv_cache.dtype == torch.bfloat16
    ):
        # The plain writer only performs norm/RoPE/BF16 storage, so applying
        # NVFP4 QDQ after it cannot be destroyed by the historical FP8 writer.
        from runtime.ds41.v41_attention_parity import qdq_compressed_cache_inplace

        qdq_compressed_cache_inplace(
            kv_cache, slot_mapping, positions, compress_ratio
        )
''',
    )

    sparse_swa = target / ".vendor/vllm-dsv41/vllm/v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        sparse_swa,
        '''        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
''',
        '''        uses_fp8_ds_mla_layout = self.dtype == torch.uint8
''',
    )

    rocm = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/amd/rocm.py"
    replace_once(
        rocm,
        "from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache\n",
        '''from vllm.models.deepseek_v4_1.common.ops import (
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
''',
    )
    old_decode = '''        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            (
                topk_ragged_indices,
                topk_ragged_indptr,
                topk_lens,
            ) = compute_global_topk_ragged_indices_and_indptr(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )

        if self.is_index_source and os.environ.get("DS41_DECODE_TRACE_DIR", ""):
            from runtime.ds41.decode_repeat_diag import get_decode_repeat_trace
            get_decode_repeat_trace().record_decode_attention(
                self.layer_id, q, swa_metadata.decode_swa_indices, swa_metadata.decode_swa_lens,
                topk_ragged_indices, topk_ragged_indptr, topk_lens,
            )

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=None,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
            extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                self.kv_cache_dtype,
                self._has_kv_transfer,
                not swa_only and kv_cache is not None,
            ),
        )
'''
    new_decode = '''        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        topk_dense_indices = None
        parity = os.environ.get("DS41_V41_ATTN_PARITY", "0") == "1"
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if parity:
                topk_dense_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )

        if self.is_index_source and os.environ.get("DS41_DECODE_TRACE_DIR", ""):
            from runtime.ds41.decode_repeat_diag import get_decode_repeat_trace
            get_decode_repeat_trace().record_decode_attention(
                self.layer_id, q, swa_metadata.decode_swa_indices, swa_metadata.decode_swa_lens,
                topk_ragged_indices, topk_ragged_indptr, topk_lens,
            )

        if parity:
            from runtime.ds41.v41_attention_parity import bf16_sparse_decode

            bf16_sparse_decode(
                q=q,
                main_cache=self.swa_cache_layer.kv_cache,
                main_indices=swa_metadata.decode_swa_indices.reshape(num_decode_tokens, -1),
                main_lens=swa_metadata.decode_swa_lens,
                extra_cache=None if swa_only else kv_cache,
                extra_indices=topk_dense_indices,
                extra_lens=topk_lens,
                attn_sink=self.attn_sink,
                scale=self.scale,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                output=output,
            )
        else:
            rocm_sparse_attn_decode(
                q=q,
                kv_cache=kv_cache,
                swa_k_cache=self.swa_cache_layer.kv_cache,
                swa_only=swa_only,
                topk_indices=None,
                topk_lens=topk_lens,
                swa_indices=swa_metadata.decode_swa_indices,
                swa_lens=swa_metadata.decode_swa_lens,
                swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
                swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
                topk_ragged_indices=topk_ragged_indices,
                topk_ragged_indptr=topk_ragged_indptr,
                attn_sink=self.attn_sink,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                output=output,
                extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                    self.kv_cache_dtype,
                    self._has_kv_transfer,
                    not swa_only and kv_cache is not None,
                ),
            )
'''
    replace_once(rocm, old_decode, new_decode)
    replace_once(
        rocm,
        '''                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )
''',
        '''                if compressed_k_cache.dtype == torch.bfloat16:
                    from runtime.ds41.v41_attention_parity import gather_bf16_paged

                    gather_bf16_paged(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                    )
                else:
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                        use_fnuz=False,
                    )
''',
    )
    replace_once(
        rocm,
        '''            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )
''',
        '''            if swa_k_cache.dtype == torch.bfloat16:
                from runtime.ds41.v41_attention_parity import gather_bf16_paged

                gather_bf16_paged(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                )
            else:
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                    use_fnuz=current_platform.is_fp8_fnuz(),
                )
''',
    )

    launcher = target / "runtime/ds41/launch-node.sh"
    replace_once(
        launcher,
        'export DS41_LOAD_PHASE_LOG=1 DS41_DROP_SHARD_CACHE=1 DS41_DROP_TENSOR_CACHE=1 DS41_GGUF_ANON_STAGE=1 DS41_STREAM_TEXT_WEIGHTS=1 DS41_MOE_C_LEGACY_TEXT_ABI=1 DS41_DECOMPOSED_QKV_INSERT=1',
        'export DS41_LOAD_PHASE_LOG=1 DS41_DROP_SHARD_CACHE=1 DS41_DROP_TENSOR_CACHE=1 DS41_GGUF_ANON_STAGE=1 DS41_STREAM_TEXT_WEIGHTS=1 DS41_MOE_C_LEGACY_TEXT_ABI=1 DS41_DECOMPOSED_QKV_INSERT=1 DS41_V41_ATTN_PARITY=1',
    )
    replace_once(
        launcher,
        'DeepSeek-V4.1-Flash-Q2-Native-M1',
        'DeepSeek-V4.1-Flash-Q2-AttnParity002-M1',
    )

    # Freeze a candidate-specific paired coordinator identity before any
    # model load.  TRANSFER 001 proved that reusing the old advertised model can
    # turn a valid rank load into HTTP400 at the control plane.
    import json

    pair_cfg_src = target / "config.transfer-ds4-native-001-l2-m1.json"
    pair_cfg_dst = target / "config.attention-parity-002-m1.json"
    pair_cfg = json.loads(pair_cfg_src.read_text())
    pair_cfg["model"] = "DeepSeek-V4.1-Flash-Q2-AttnParity002-M1"
    pair_cfg["profile_status"] = "DS41_V41_ATTENTION_PARITY_002_M1"
    pair_cfg_dst.write_text(json.dumps(pair_cfg, indent=2, sort_keys=True) + "\n")

    marker = target / "runtime/ds41/attention-parity-002-release.txt"
    marker.write_text(
        "DS41_V41_ATTENTION_PARITY_002\n"
        "base=native-antirez-m1-transfer001-woa1\n"
        "window=full512-fp8-block32-ue8m0-qdq-bf16-depot\n"
        "compressed=full512-nvfp4-block16-e4m3-qdq-bf16-depot\n"
        "indexer=mxfp4-block32-ue8m0-packed-rowmajor-rocm-compat-consumer\n"
    )

    print("ATTN_PARITY_002_PATCHED")


if __name__ == "__main__":
    main()
