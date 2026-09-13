#!/usr/bin/env python3
"""CPU-only checks of diagnostic boundaries and transparent reader hooks."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from runtime.ds41.affine_safetensors import AffineRowLRU, SafeTensorMMap
from runtime.ds41.fault_diagnostics import FaultDiagnostics, process_snapshot


class Processor:
    def __init__(self):
        self.calls = []

    def process_outputs(self, outputs, engine_core_timestamp=None, iteration_stats=None):
        self.calls.append((outputs, engine_core_timestamp, iteration_stats))
        return SimpleNamespace(request_outputs=[item for item in outputs if item.finished])


def output(tokens, finished=False):
    return SimpleNamespace(request_id="request", new_token_ids=tokens, finished=finished)


def fixture(path):
    header, blobs, offset = {}, [], 0
    for suffix, dtype, data in (
        ("weight", "U32", np.zeros((4, 4), dtype="<u4")),
        ("scales", "BF16", np.full((4, 1), 0x3f80, dtype="<u2")),
        ("biases", "BF16", np.zeros((4, 1), dtype="<u2")),
    ):
        raw = data.tobytes()
        header["layer.embed." + suffix] = {
            "dtype": dtype, "shape": list(data.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    raw_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs))


class DiagnosticsTest(unittest.TestCase):
    def test_boundaries_cache_counts_and_reader_transparency(self):
        original = SafeTensorMMap.affine2_rows
        processor = Processor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.safetensors"
            fixture(path)
            source = SafeTensorMMap(path)
            cache = AffineRowLRU(source, "layer.embed", max_rows=4)
            second_cache = AffineRowLRU(source, "layer.embed", max_rows=4)
            expected = source.affine2_rows("layer.embed", [0, 1])
            collector = FaultDiagnostics(processor, directory, rank=0)
            with collector, patch.object(collector, "_snapshot", wraps=collector._snapshot) as snap:
                collector.begin_request("P")
                got = cache.lookup(np.array([0, 1]))
                np.testing.assert_array_equal(got, expected)
                second_cache.lookup(np.array([2]))
                batch = [output([11])]
                marker = object()
                processor.process_outputs(batch, 7.0, iteration_stats=marker)
                self.assertIs(processor.calls[-1][0], batch)
                self.assertIs(processor.calls[-1][2], marker)
                for token in (12, 13, 14):
                    processor.process_outputs([output([token])])
                self.assertEqual(snap.call_count, 2)
                cache.lookup(np.array([0, 2]))
                processor.process_outputs([output([15], finished=True)])
                self.assertEqual(snap.call_count, 3)
                result = collector.end_request()
                self.assertEqual(snap.call_count, 4)
                self.assertEqual(result["observed_tokens"], 5)
                phases = {bucket["phase"]: bucket for bucket in result["reader"]}
                self.assertEqual(phases["before_first"]["requested_rows"], 3)
                self.assertEqual(phases["decode"]["requested_rows"], 1)
                self.assertEqual(phases["before_first"]["logical_source_bytes"], 60)
                self.assertIn("ru_utime", phases["decode"]["rusage_thread"])
                self.assertEqual(result["start"]["engram_cache"][str(id(cache))]["rows_read"], 0)
                self.assertEqual(result["intervals"]["request_to_first"]["engram_cache_delta"][str(id(cache))]["rows_read"], 2)
                self.assertGreater(result["instrumentation_overhead_ns"], 0)
                json.dumps(result)
                maps = collector.snapshot_maps("after_P")
                data = json.loads(Path(maps["path"]).read_text())
                self.assertTrue(data["regions"])
                self.assertTrue(data["by_path_kib"])
            self.assertIs(SafeTensorMMap.affine2_rows, original)
            self.assertNotIn("process_outputs", vars(processor))
            source.close()

    def test_single_token_and_empty_terminal_batch(self):
        processor = Processor()
        with tempfile.TemporaryDirectory() as directory, FaultDiagnostics(processor, directory) as collector:
            collector.begin_request("single")
            with self.assertRaises(RuntimeError):
                collector.snapshot_maps("during_request")
            processor.process_outputs([output([1], True)])
            result = collector.end_request()
            self.assertEqual(result["intervals"]["first_to_last"]["wall_ns"], 0)
            collector.begin_request("empty_terminal")
            processor.process_outputs([output([1])])
            processor.process_outputs([output([], True)])
            result = collector.end_request()
            self.assertFalse(result["last_token_snapshot_available"])
            self.assertNotIn("first_to_last", result["intervals"])

    def test_restore_on_exception_and_read_errors(self):
        original = AffineRowLRU.lookup
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sentinel"):
                with FaultDiagnostics(Processor(), directory):
                    raise ValueError("sentinel")
        self.assertIs(AffineRowLRU.lookup, original)
        with patch("pathlib.Path.read_text", side_effect=PermissionError("denied")):
            snapshot = process_snapshot()
        self.assertTrue(snapshot["read_errors"])
        self.assertEqual(snapshot["process_io"], {})
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
