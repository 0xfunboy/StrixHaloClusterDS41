#!/usr/bin/env python3
"""Model-free DS41 vocab/embed/head loader identity test on the real GGUF shard5."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from _ds41_artifact import MODEL_DIR

import torch
from torch.nn.parameter import UninitializedParameter

from vllm_gguf_plugin.quantization.params import _gguf_embedding_weight_loader
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
MODEL = MODEL_DIR
SHARD = MODEL / 'DSV41-mixedq2-00005-of-00005.gguf'
ATTEMPT = ROOT / 'reports/DS41-Q2-001/attempt010'
VOCAB = 129280
HIDDEN = 5120
BOUNDARY = VOCAB // 2


def sha_row(t: torch.Tensor) -> str:
    # Preserve exact BF16 bits.
    return hashlib.sha256(t.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def load_attempt_ids() -> tuple[list[int], list[int]]:
    prompt = json.loads((ATTEMPT / 'prompt-tokens.json').read_text())['prompts']['arithmetic']['token_ids']
    result_path = ATTEMPT / 'offline-rank0.json'
    if not result_path.exists():
        result_path = ATTEMPT / 'offline-rank1.json'
    result = json.loads(result_path.read_text())
    output = next(x for x in result['results'] if x['label'] == 'smoke-arithmetic')['token_ids']
    return prompt, output


def make_param(start: int, end: int) -> tuple[SimpleNamespace, UninitializedParameter]:
    layer = SimpleNamespace(
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=end,
        )
    )
    p = UninitializedParameter(requires_grad=False, device='cpu')
    p.output_dim = 0
    p.input_dim = 1
    p.tensor_shape = (end - start, HIDDEN)
    return layer, p


def check_vocab_tensor(kind: str, source: torch.Tensor, ids: list[int], hidden: torch.Tensor | None):
    assert source.dtype == torch.bfloat16, (kind, source.dtype)
    assert tuple(source.shape) == (VOCAB, HIDDEN), (kind, source.shape)
    report = {'source_shape': list(source.shape), 'source_dtype': str(source.dtype), 'ranks': {}}
    distributed_logits: dict[int, float] = {}
    source_logits: dict[int, float] = {}

    for rank, (start, end) in enumerate(((0, BOUNDARY), (BOUNDARY, VOCAB))):
        layer, p = make_param(start, end)
        _gguf_embedding_weight_loader(layer, p, source)
        assert tuple(p.shape) == (BOUNDARY, HIDDEN), (kind, rank, p.shape)
        owned = [i for i in ids if start <= i < end]
        rows = []
        for gid in owned:
            local = gid - start
            got = p[local].detach().cpu()
            ref = source[gid].detach().cpu()
            exact = torch.equal(got, ref)
            if not exact:
                raise AssertionError(f'{kind} rank{rank} row {gid} mismatch')
            item = {
                'global_id': gid,
                'local_id': local,
                'exact': exact,
                'sha256': sha_row(got),
            }
            if hidden is not None:
                # Independent sampled projection: source row and final loader row
                # are accumulated in fp32 using the same fixed hidden vector.
                a = torch.dot(got.float(), hidden).item()
                b = torch.dot(ref.float(), hidden).item()
                item['loaded_logit_fp32'] = a
                item['source_logit_fp32'] = b
                item['logit_abs'] = abs(a - b)
                distributed_logits[gid] = a
                source_logits[gid] = b
            rows.append(item)
        report['ranks'][str(rank)] = {
            'global_range': [start, end],
            'final_shape': list(p.shape),
            'rows': rows,
        }
        del p, layer
        gc.collect()

    if hidden is not None:
        ordered = sorted(distributed_logits)
        max_abs = max((abs(distributed_logits[i] - source_logits[i]) for i in ordered), default=0.0)
        report['sample_projection'] = {
            'global_ids': ordered,
            'max_abs': max_abs,
            'all_exact': max_abs == 0.0,
        }
        if max_abs != 0.0:
            raise AssertionError(f'{kind} sampled distributed projection mismatch {max_abs}')
    return report


def main():
    prompt_ids, output_ids = load_attempt_ids()
    ids = sorted(set(
        [0, 1, 2, BOUNDARY - 1, BOUNDARY, BOUNDARY + 1, VOCAB - 1]
        + prompt_ids
        + output_ids[:32]
    ))
    # Fixed reference hidden, deterministic and bounded.
    hidden = (((torch.arange(HIDDEN, dtype=torch.float32) % 257) - 128.0) / 128.0).contiguous()

    out = {
        'status': 'IN_PROGRESS',
        'vocab_size': VOCAB,
        'hidden_size': HIDDEN,
        'tp_boundary': BOUNDARY,
        'selected_global_ids': ids,
        'prompt_ids': prompt_ids,
        'attempt010_output_ids_head': output_ids[:32],
        'mapping': {
            'token_embd': 'embed.weight -> language_model.model.embed_tokens.weight',
            'output': 'head.weight -> language_model.lm_head.weight',
            'output_norm': 'norm.weight -> language_model.model.norm.weight',
        },
    }

    name_map = {
        'token_embd': 'embed.weight',
        'output': 'head.weight',
        'output_norm': 'norm.weight',
    }
    seen = set()
    for name, tensor in gguf_quant_weights_iterator_multi([str(SHARD)], name_map):
        if name == 'embed.weight':
            out['embedding'] = check_vocab_tensor('embedding', tensor, ids, None)
            seen.add(name)
        elif name == 'head.weight':
            out['lm_head'] = check_vocab_tensor('lm_head', tensor, ids, hidden)
            seen.add(name)
        elif name == 'norm.weight':
            assert tensor.dtype == torch.bfloat16
            assert tuple(tensor.shape) == (HIDDEN,)
            # This parameter is replicated/unsharded; default loader semantics
            # are an exact copy. Record the full BF16 bit hash and finite stats.
            bits = tensor.contiguous().view(torch.uint16).numpy().tobytes()
            vals = tensor.float()
            out['output_norm'] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'sha256_bf16_bits': hashlib.sha256(bits).hexdigest(),
                'finite': bool(torch.isfinite(vals).all().item()),
                'min': vals.min().item(),
                'max': vals.max().item(),
                'mean': vals.mean().item(),
            }
            seen.add(name)
    expected = {'embed.weight','head.weight','norm.weight'}
    if seen != expected:
        raise AssertionError(f'missing final tensors: seen={seen}')
    out['status'] = 'PASS'
    outpath = ROOT / 'reports/DS41-Q2-001/stage0/vocab-head-real-node01.json'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({
        'status': out['status'],
        'embedding_rows_checked': sum(len(x['rows']) for x in out['embedding']['ranks'].values()),
        'head_rows_checked': sum(len(x['rows']) for x in out['lm_head']['ranks'].values()),
        'head_projection_max_abs': out['lm_head']['sample_projection']['max_abs'],
        'norm': out['output_norm'],
        'report': str(outpath),
    }, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
