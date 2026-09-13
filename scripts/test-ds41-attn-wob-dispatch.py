#!/usr/bin/env python3
"""Model-free M=1 dispatch gate for the DS41 WO_B LLMM1 helper.

A reduction spy checks exactly one call and propagation of its return value.
This gate does not run a distributed collective. The separate LLMM1 gate checks
arithmetic sums of both DenseFix shards, also without distributed execution.
Set DS41_WOB_DISPATCH_OUT to a new report path to preserve previous raw results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm.distributed as dist_mod
from vllm import _custom_ops as ops
from vllm import envs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from runtime.ds41.attn_wob_llmm1 import reset_stats, stats, try_wob_llmm1

REL_MAX = 5e-3
ABS_MAX = 1.25e-1


def metric(a, b):
    af, bf = a.float(), b.float()
    d = af - bf
    return {
        'max_abs': float(d.abs().max()),
        'rel_l2': float(d.norm()) / max(float(bf.norm()), 1e-30),
        'finite': bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
        'exact_equal': bool(torch.equal(a, b)),
    }


def check_metric(name, value):
    if not value['finite'] or value['rel_l2'] > REL_MAX or value['max_abs'] > ABS_MAX:
        raise RuntimeError(f'{name}: {value}')


def main():
    if not os.environ.get('DS41_WOB_DISPATCH_OUT'):
        raise RuntimeError('Set DS41_WOB_DISPATCH_OUT to a new report path.')
    out_path = Path(os.environ['DS41_WOB_DISPATCH_OUT'])
    if out_path.exists():
        raise FileExistsError(f'Preserving existing raw report: {out_path}')
    assert torch.cuda.is_available() and torch.version.hip is not None
    g = torch.Generator(device='cuda').manual_seed(8842)
    x = torch.randn((1, 4096), device='cuda', dtype=torch.bfloat16, generator=g)
    w = (torch.randn((5120, 4096), device='cuda', dtype=torch.bfloat16, generator=g) * 0.005).contiguous()
    layer = SimpleNamespace(
        weight=w, bias=None, input_is_parallel=True, reduce_results=True,
        tp_size=2, return_bias=False, quant_method=UnquantizedLinearMethod(),
    )
    ref = layer.quant_method.apply(layer, x, None)
    baseline_equivalence = metric(ref, torch.nn.functional.linear(x, w))
    check_metric('baseline vs F.linear', baseline_equivalence)
    kernel_calls = []
    reduction_calls = []
    fallback_cases = []
    original_llmm1 = ops.LLMM1

    def kernel_spy(weight, activation, rows_per_block):
        assert weight is w and activation is x and rows_per_block == 4
        result = original_llmm1(weight, activation, rows_per_block)
        kernel_calls.append(result)
        return result

    def reduction_spy(local):
        assert len(kernel_calls) == 1 and local is kernel_calls[0]
        # Return a distinct tensor to detect accidentally returning the local result.
        reduced = local.clone()
        reduction_calls.append((local, reduced))
        return reduced

    def changed_layer(**changes):
        return SimpleNamespace(**(vars(layer) | changes))

    def expect_fallback(name, reason, candidate_layer=layer, activation=x):
        before = stats()
        call_counts = (len(kernel_calls), len(reduction_calls))
        assert try_wob_llmm1(candidate_layer, activation) is None, name
        after = stats()
        assert (len(kernel_calls), len(reduction_calls)) == call_counts, name
        assert after['llmm1_calls'] == before['llmm1_calls'], name
        assert after['llmm1_tokens'] == before['llmm1_tokens'], name
        assert after['fallback_calls'] == before['fallback_calls'] + 1, name
        assert after['fallback_reasons'].get(reason, 0) == before['fallback_reasons'].get(reason, 0) + 1, (name, after)
        fallback_cases.append({'case': name, 'reason': reason, 'kernel_calls': 0, 'reduction_calls': 0})

    with (
        patch.dict(os.environ, {'DS41_ATTN_WOB_LLMM1': '1'}),
        patch.object(ops, 'LLMM1', kernel_spy),
        patch.object(dist_mod, 'tensor_model_parallel_all_reduce', reduction_spy),
    ):
        reset_stats()
        cand = try_wob_llmm1(layer, x)
        assert len(kernel_calls) == len(reduction_calls) == 1
        assert cand is reduction_calls[0][1]
        m = metric(cand, ref)
        check_metric('LLMM1 vs configured baseline', m)

        expect_fallback('M=2', 'tokens_not_1', activation=x.expand(2, -1).contiguous())
        expect_fallback('M=0', 'tokens_not_1', activation=x[:0])
        expect_fallback('wrong rank', 'tokens_not_1', activation=x.flatten())
        expect_fallback('non-tensor input', 'tokens_not_1', activation=None)
        expect_fallback('input dtype', 'input_contract', activation=x.float())
        noncontiguous_x = torch.empty((1, 8192), device=x.device, dtype=x.dtype)[:, ::2]
        expect_fallback('input strides', 'input_contract', activation=noncontiguous_x)
        expect_fallback('missing weight', 'missing_weight', changed_layer(weight=None))
        expect_fallback('CPU input', 'device_contract', activation=x.cpu())
        expect_fallback('CPU weight', 'device_contract', changed_layer(weight=torch.empty_like(w, device='cpu')))
        expect_fallback('weight dtype', 'weight_contract', changed_layer(weight=w.view(torch.int16)))
        expect_fallback('weight strides', 'weight_contract', changed_layer(weight=w.as_strided(w.shape, (1, 5120))))
        expect_fallback('input K', 'shape_contract', activation=x[:, :2048])
        expect_fallback('weight shape', 'shape_contract', changed_layer(weight=w[:2560]))
        expect_fallback('bias', 'bias_present', changed_layer(bias=x[0, :1]))
        expect_fallback('tuple return', 'return_bias', changed_layer(return_bias=True))
        no_return_bias = SimpleNamespace(**{k: v for k, v in vars(layer).items() if k != 'return_bias'})
        expect_fallback('missing return contract', 'return_bias', no_return_bias)
        expect_fallback('unsharded input', 'input_not_parallel', changed_layer(input_is_parallel=False))
        expect_fallback('no reduction', 'tp_contract', changed_layer(reduce_results=False))
        expect_fallback('TP1', 'tp_contract', changed_layer(tp_size=1))
        expect_fallback('TP4', 'tp_contract', changed_layer(tp_size=4))
        expect_fallback('missing quant method', 'quant_method', changed_layer(quant_method=None))
        lookalike = type('UnquantizedLinearMethod', (), {})()
        expect_fallback('quant method name collision', 'quant_method', changed_layer(quant_method=lookalike))
        with patch.object(torch.version, 'hip', None):
            expect_fallback('non-ROCm backend', 'backend_not_rocm')
        with patch.object(envs, 'VLLM_BATCH_INVARIANT', True):
            expect_fallback('batch invariance', 'batch_invariant')
        with patch.object(torch.ops, '_rocm_C', SimpleNamespace()):
            expect_fallback('operator unavailable', 'llmm1_unavailable')

        for flag in (None, '0', 'true'):
            if flag is None:
                os.environ.pop('DS41_ATTN_WOB_LLMM1', None)
            else:
                os.environ['DS41_ATTN_WOB_LLMM1'] = flag
            before = stats()
            assert try_wob_llmm1(layer, x) is None
            assert stats() == before
            assert len(kernel_calls) == len(reduction_calls) == 1
        st = stats()
        assert st['llmm1_calls'] == st['llmm1_tokens'] == 1
        assert st['fallback_calls'] == len(fallback_cases)

    out = {
        'status': 'PASS', 'metric': m, 'stats': st,
        'gate': {'rel_l2_max': REL_MAX, 'max_abs': ABS_MAX},
        'baseline': 'UnquantizedLinearMethod.apply(layer, x, None)',
        'baseline_vs_f_linear': baseline_equivalence,
        'qualified_shape': {'x': [1, 4096], 'weight': [5120, 4096], 'dtype': 'bfloat16'},
        'kernel_calls': len(kernel_calls), 'reduction_calls': len(reduction_calls),
        'reduction_return_propagated': True, 'real_collective_tested': False,
        'fallback_cases': fallback_cases, 'disabled_flags_checked': [None, '0', 'true'],
        'scope': 'M=1 shape, including possible one-token prefill or dummy calls',
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x') as handle:
        handle.write(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
