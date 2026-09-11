#!/usr/bin/env python3
"""Prove the DS41 GGUF iterator exposes the large routed tensor as an mmap view."""
from __future__ import annotations

import gc
from pathlib import Path

from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi

MODEL = Path('/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2/DSV41-mixedq2-00001-of-00005.gguf')
RAW = 'blk.0.ffn_gate_exps'
MAPPED = 'layers.0.ffn.experts.0.w1.weight'


def anon_kib() -> int:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('RssAnon:'):
            return int(line.split()[1])
    raise RuntimeError('RssAnon missing')


def main() -> None:
    gc.collect()
    before = anon_kib()
    got = None
    it = gguf_quant_weights_iterator_multi([str(MODEL)], {RAW: MAPPED})
    for name, tensor in it:
        if name == MAPPED:
            got = tensor
            break
    assert got is not None
    # This tensor is >1 GiB logically; a clone would raise RssAnon by roughly that amount.
    logical = got.numel() * got.element_size()
    after = anon_kib()
    delta = max(0, after - before) * 1024
    assert logical > 1_000_000_000, logical
    assert delta < 128 * 1024 * 1024, (logical, delta)
    assert got.device.type == 'cpu' and str(got.dtype) == 'torch.uint8'
    print({
        'status': 'PASS',
        'tensor': RAW,
        'shape': tuple(got.shape),
        'logical_bytes': logical,
        'rss_anon_delta_bytes': delta,
        'ratio': delta / logical,
    })


if __name__ == '__main__':
    main()
