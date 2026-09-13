#!/usr/bin/env python3
"""CPU checks of matched Engram preparation using only synthetic local files."""
from __future__ import annotations

import importlib.util
import mmap
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.ds41.affine_safetensors import AffineRowLRU, SafeTensorMMap
from runtime.ds41.engram_advice_experiment import prepare_embedding, resident_pages

spec = importlib.util.spec_from_file_location(
    "ds41_advice_fixture", Path(__file__).with_name("test-ds41-engram-advice.py"))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
BASE = fixture_module.BASE


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        # /tmp is tmpfs here, where file-cache discard cannot produce a cold
        # disk-backed mapping. Use a temporary directory on the workspace disk.
        self.directory = tempfile.TemporaryDirectory(
            prefix=".engram-experiment-test-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.safetensors"
        fixture_module.fixture(self.path)
        # Production artifacts are already clean. Match that condition on this
        # newly written synthetic fixture before testing discard effectiveness.
        with self.path.open("r+b") as stream:
            os.fsync(stream.fileno())
        source = SafeTensorMMap(self.path)
        with patch.dict(os.environ, {"DS41_ENGRAM_RANDOM_ADVICE": "0"}):
            cache = AffineRowLRU(source, BASE, max_rows=65536)
        self.embedding = SimpleNamespace(source=source, cache=cache, base=BASE)
        self.addCleanup(lambda: self.embedding.source.close())

    def test_fresh_reader_scoped_zero_residency_and_unchanged_rows(self):
        sidecar = SafeTensorMMap(self.path)
        self.addCleanup(sidecar.close)
        sidecar_fd = sidecar._fd
        q_before = sidecar.array("layers.1.engram.q").copy()
        embedding = self.embedding
        cache = embedding.cache
        expected = cache.lookup(np.array([0, 4095, 8191]))
        cache.lookup(np.array([0]))
        counters = (cache.hits, cache.misses, cache.rows_read)
        for policy in ("random", "normal"):
            old = embedding.source
            old_fd = old._fd
            old_mapping = old._mmap
            cached_rows = len(cache._rows)
            with patch("runtime.ds41.engram_advice_experiment.os.posix_fadvise",
                       wraps=os.posix_fadvise) as discard:
                record = prepare_embedding(embedding, policy)
            discard.assert_called_once_with(
                embedding.source._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            self.assertEqual(record["discard_scope"], "whole_identified_engram_file_test_only")
            self.assertEqual(record["discard_mapping_count"], 2)
            self.assertEqual(record["discard_file_identity"]["st_ino"], self.path.stat().st_ino)
            self.assertIsNot(embedding.source, old)
            self.assertIs(cache.source, embedding.source)
            self.assertEqual(record["old_fd"], old_fd)
            self.assertNotEqual(record["new_fd"], old_fd)
            self.assertEqual(old._fd, -1)
            self.assertTrue(old_mapping.closed)
            self.assertEqual(record["cleared_decoded_rows"], cached_rows)
            self.assertEqual(len(cache._rows), 0)
            self.assertEqual(cache.max_rows, 65536)
            self.assertEqual((cache.hits, cache.misses, cache.rows_read), counters)
            self.assertEqual(cache.advice_status["effective"], policy)
            self.assertEqual(len(record["ranges"]), 3)
            for bounds, observed in zip(record["advice"]["ranges"], record["ranges"]):
                start, length = observed["start"], observed["length"]
                self.assertEqual(start % mmap.PAGESIZE, 0)
                self.assertEqual(length % mmap.PAGESIZE, 0)
                self.assertGreaterEqual(start, bounds["tensor_start"])
                self.assertLessEqual(start + length, bounds["tensor_end_exclusive"])
                self.assertEqual(bounds["unadvised_boundary_bytes"],
                                 bounds["tensor_end_exclusive"] - bounds["tensor_start"] - length)
                self.assertLess(bounds["unadvised_boundary_bytes"], 2 * mmap.PAGESIZE)
                self.assertEqual(observed["after"]["resident_pages"], 0)
                self.assertEqual(resident_pages(embedding.source, start, length)["resident_pages"], 0)
            np.testing.assert_array_equal(embedding.source.affine2_rows(BASE, [0, 4095, 8191]), expected)
            self.assertEqual((cache.hits, cache.misses, cache.rows_read), counters)
            self.assertEqual(sidecar._fd, sidecar_fd)
            self.assertFalse(sidecar._mmap.closed)
            self.assertEqual(sidecar.engram_advice, {})
            np.testing.assert_array_equal(sidecar.array("layers.1.engram.q"), q_before)
        self.assertNotIn("torch", sys.modules)

    def test_interior_only_discard_reproducer_still_fails_closed(self):
        real_discard = os.posix_fadvise

        def old_interior_discard(fd, start, length, option):
            self.assertEqual((start, length, option), (0, 0, os.POSIX_FADV_DONTNEED))
            for bounds in self.embedding.cache.advice_status["ranges"]:
                real_discard(fd, bounds["advice_start"], bounds["advice_length"], option)

        # This ext4 synthetic case reproduced retained boundary folios despite
        # aligned tensor-interior discard. The strict gate must reject them.
        with patch("runtime.ds41.engram_advice_experiment.os.posix_fadvise",
                   side_effect=old_interior_discard):
            with self.assertRaisesRegex(RuntimeError, "nonzero initial table residency"):
                prepare_embedding(self.embedding, "random")

    def test_nonzero_residency_rejects_preparation(self):
        def resident(source, start, length):
            count = length // mmap.PAGESIZE
            return {"pages": count, "resident_pages": 1, "vector_bytes": count}

        with patch("runtime.ds41.engram_advice_experiment.resident_pages", side_effect=resident) as probe:
            with self.assertRaisesRegex(RuntimeError, "nonzero initial table residency"):
                prepare_embedding(self.embedding, "random")
        self.assertGreaterEqual(probe.call_count, 2)
        # A failed gate cannot produce a success record, but the retained fresh
        # reader remains valid for normal process cleanup.
        self.assertIs(self.embedding.cache.source, self.embedding.source)
        self.assertFalse(self.embedding.source._mmap.closed)


if __name__ == "__main__":
    unittest.main()
