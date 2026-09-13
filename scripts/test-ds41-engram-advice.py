#!/usr/bin/env python3
"""CPU-only synthetic checks of scoped Engram mapping advice."""
from __future__ import annotations

import json
import mmap
from dataclasses import replace
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.ds41.affine_safetensors import AffineRowLRU, SafeTensorMMap


BASE = "layers.1.engram.embed"


def fixture(path, *, base=BASE, rows=8192, overlap=False):
    specs = [
        ("layers.1.engram.q", "F32", np.zeros(257, dtype="<f4")),
        (base + ".weight", "U32", np.tile(np.array([0, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF], dtype="<u4"), (rows, 1))),
        (base + ".scales", "BF16", np.full((rows, 1), 0x3F80, dtype="<u2")),
        (base + ".biases", "BF16", np.full((rows, 1), 0x4000, dtype="<u2")),
        ("layers.1.engram.wkv", "F32", np.zeros(257, dtype="<f4")),
    ]
    header, blobs, offset = {}, [], 0
    for name, dtype, values in specs:
        raw = values.tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if overlap:
        header["layers.1.engram.q"]["data_offsets"] = [1028, 2056]
    encoded = json.dumps(header).encode().ljust(mmap.PAGESIZE, b" ")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


class MappingProxy:
    """Allow syscall failures without replacing the actual mapping buffer."""

    def __init__(self, actual, fail_calls=()):
        self.actual = actual
        self.fail_calls = set(fail_calls)
        self.calls = []

    @property
    def closed(self):
        return self.actual.closed

    def __len__(self):
        return len(self.actual)

    def madvise(self, option, start, length):
        self.calls.append((option, start, length))
        if len(self.calls) in self.fail_calls:
            raise OSError("synthetic advisory failure")
        self.actual.madvise(option, start, length)


class AdviceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.safetensors"
        fixture(self.path)
        self.source = SafeTensorMMap(self.path)
        self.addCleanup(self.source.close)

    def test_exact_rows_default_random_normal_and_cache_preservation(self):
        expected = np.repeat(np.array([2, 3, 4, 5], dtype=np.float32), 16)
        with patch.dict("os.environ", {"DS41_ENGRAM_RANDOM_ADVICE": "0"}):
            cache = AffineRowLRU(self.source, BASE, max_rows=4)
        self.assertEqual(cache.advice_status["status"], "not_requested")
        first = cache.lookup(np.array([0, 8191]))
        np.testing.assert_array_equal(first, np.tile(expected, (2, 1)))
        cached = list(cache._rows.items())
        original_bytes = self.path.read_bytes()
        for policy in ("random", "normal"):
            result = self.source.set_engram_advice(BASE, policy)
            self.assertEqual((result["status"], result["effective"]), ("applied", policy))
            self.assertIs(cache.advice_status, result)
            np.testing.assert_array_equal(self.source.affine2_rows(BASE, [0, 8191]), first)
            self.assertEqual(list(cache._rows), [key for key, _ in cached])
            for key, value in cached:
                self.assertIs(cache._rows[key], value)
            self.assertEqual((cache.max_rows, cache.hits, cache.misses, cache.rows_read), (4, 0, 2, 2))
        self.assertEqual(self.path.read_bytes(), original_bytes)

    def test_opt_in_and_complete_page_bounds(self):
        with patch.dict("os.environ", {"DS41_ENGRAM_RANDOM_ADVICE": "1"}):
            cache = AffineRowLRU(self.source, BASE, max_rows=4)
        self.assertEqual(cache.advice_status["effective"], "random")
        for region in cache.advice_status["ranges"]:
            start, length = region["advice_start"], region["advice_length"]
            self.assertEqual(start % mmap.PAGESIZE, 0)
            self.assertEqual(length % mmap.PAGESIZE, 0)
            self.assertGreaterEqual(start, region["tensor_start"])
            self.assertLessEqual(start + length, region["tensor_end_exclusive"])
            self.assertGreater(region["unadvised_boundary_bytes"], 0)
        self.assertNotIn("torch", sys.modules)

    def test_failure_rollback_and_unknown_state(self):
        for failures, expected in (({2}, ("fallback_normal", "normal")),
                                   ({2, 3}, ("failed", "mixed_or_unknown"))):
            proxy = MappingProxy(self.source._mmap, failures)
            with patch.object(self.source, "_mmap", proxy):
                result = self.source.set_engram_advice(BASE, "random")
            self.assertEqual((result["status"], result["effective"]), expected)
            self.assertTrue(result["errors"])
            self.assertEqual(len(proxy.calls), 5)
        np.testing.assert_array_equal(self.source.affine2_rows(BASE, [0]), self.source.affine2_rows(BASE, [8191]))
        self.assertEqual(self.source.set_engram_advice(BASE, "normal")["effective"], "normal")

    def test_unsupported_and_default_fallback(self):
        with patch.object(mmap, "MADV_RANDOM", None), patch.dict("os.environ", {"DS41_ENGRAM_RANDOM_ADVICE": "1"}):
            cache = AffineRowLRU(self.source, BASE, max_rows=4)
        self.assertEqual((cache.advice_status["status"], cache.advice_status["effective"]), ("unsupported", "normal"))
        self.assertTrue(cache.advice_status["errors"])
        self.assertEqual(cache.lookup(np.array([0])).shape, (1, 64))
        proxy = MappingProxy(self.source._mmap)
        proxy.madvise = None
        with patch.object(self.source, "_mmap", proxy):
            self.assertEqual(self.source.set_engram_advice(BASE, "random")["status"], "unsupported")

    def test_guards_and_small_regions(self):
        proxy = MappingProxy(self.source._mmap)
        with patch.object(self.source, "_mmap", proxy):
            for base, policy in (("layers.1.engram.q", "random"), ("layers.1.engram.k", "random"),
                                 ("layers.1.engram.wkv", "random"), ("layers.2.engram.embed", "random"),
                                 (BASE, "willneed"), ("layers.14.engram.embed", "random")):
                self.assertEqual(self.source.set_engram_advice(base, policy)["status"], "rejected")
            with patch.object(self.source, "path", self.path.with_suffix(".gguf")):
                self.assertEqual(self.source.set_engram_advice(BASE, "random")["status"], "rejected")
            with patch.dict("os.environ", {"DS41_ENGRAM_RANDOM_ADVICE": "1"}):
                cache = AffineRowLRU(self.source, "unapproved.embed")
            self.assertEqual(cache.advice_status["status"], "rejected")
            self.assertTrue(cache.advice_status["errors"])
        self.assertEqual(proxy.calls, [])
        for kwargs, expected in (({"rows": 1}, "no_complete_pages"), ({"overlap": True}, "rejected"),
                                 ({"base": "layers.14.engram.embed"}, "applied")):
            path = Path(self.directory.name) / "other.safetensors"
            fixture(path, **kwargs)
            source = SafeTensorMMap(path)
            try:
                result = source.set_engram_advice(kwargs.get("base", BASE), "random")
                self.assertEqual(result["status"], expected)
            finally:
                source.close()
        self.source.close()
        self.assertEqual(self.source.set_engram_advice(BASE, "normal")["status"], "rejected")

    def test_invalid_affine_shapes_and_offsets_do_not_issue_advice(self):
        name = BASE + ".scales"
        entry = self.source.entries[name]
        proxy = MappingProxy(self.source._mmap)
        with patch.object(self.source, "_mmap", proxy):
            for bad in (replace(entry, shape=(8192, 2)), replace(entry, dtype="F32"),
                        replace(entry, offset=0), replace(entry, offset=len(proxy))):
                with patch.dict(self.source.entries, {name: bad}):
                    result = self.source.set_engram_advice(BASE, "random")
                self.assertEqual(result["status"], "rejected")
                self.assertTrue(result["errors"])
        self.assertEqual(proxy.calls, [])


if __name__ == "__main__":
    unittest.main()
