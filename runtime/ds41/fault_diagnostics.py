"""Opt-in CPU observations for one offline SPMD request at a time.

Counters include instrumentation. Reader elapsed time includes CPU work and is
not an I/O-wait measurement. Logical source bytes are not physical disk bytes.
RUSAGE_CHILDREN covers reaped children only; live child work is not attributed.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
import resource
import threading
import time
import weakref


_STATUS = {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "Threads"}
_MEMINFO = {"MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree", "SwapCached"}
_VMSTAT = {"pswpin", "pswpout", "pgmajfault"}
_RU = ("ru_minflt", "ru_majflt", "ru_inblock", "ru_oublock", "ru_utime", "ru_stime")


def _usage(who):
    value = resource.getrusage(who)
    return {key: getattr(value, key) for key in _RU}


def _delta(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def _read(path, errors):
    try:
        return Path(path).read_text()
    except OSError as exc:
        errors[str(path)] = str(exc)
        return ""


def _numbers(text, selected=None):
    values = {}
    for line in text.splitlines():
        fields = line.replace(":", " ").split()
        if len(fields) > 1 and (selected is None or fields[0] in selected):
            try:
                values[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return values


def process_snapshot(rank=None):
    """Read this process and host counters, without inspecting GPU state."""
    entered = time.perf_counter_ns()
    errors = {}
    children = set()
    for path in Path("/proc/self/task").glob("*/children"):
        children.update(int(pid) for pid in _read(path, errors).split())
    result = {
        "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(),
        "ppid": os.getppid(), "tid": threading.get_native_id(), "rank": rank,
        "rusage_self": _usage(resource.RUSAGE_SELF),
        "rusage_children": _usage(resource.RUSAGE_CHILDREN),
        "observed_live_child_pids": sorted(children),
        "process_io": _numbers(_read("/proc/self/io", errors)),
        "status_kib_except_Threads": _numbers(_read("/proc/self/status", errors), _STATUS),
        "meminfo_kib": _numbers(_read("/proc/meminfo", errors), _MEMINFO),
        "global_vmstat": _numbers(_read("/proc/vmstat", errors), _VMSTAT),
        "read_errors": errors,
    }
    result["snapshot_overhead_ns"] = time.perf_counter_ns() - entered
    return result


class FaultDiagnostics:
    def __init__(self, output_processor, output_dir, rank=None):
        self.processor = output_processor
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.active = None
        self.overhead_ns = 0
        self.map_snapshots = []
        self._patches = []
        self._caches = weakref.WeakKeyDictionary()

    def _cache_stats(self):
        return {str(id(cache)): {
            "path": str(cache.source.path), "base": cache.base,
            "thread_ids": sorted(tids), "max_rows": cache.max_rows,
            **{key: getattr(cache, key) for key in ("hits", "misses", "rows_read")},
        } for cache, tids in list(self._caches.items())}

    def _snapshot(self):
        result = process_snapshot(self.rank)
        result["engram_cache"] = self._cache_stats()
        return result

    def _replace(self, target, name, replacement):
        had_own = name in vars(target)
        original = getattr(target, name)
        self._patches.append((target, name, original, had_own, replacement))
        setattr(target, name, replacement)

    def install(self):
        if self._patches:
            return self
        from runtime.ds41.affine_safetensors import AffineRowLRU, SafeTensorMMap

        original_outputs = self.processor.process_outputs
        original_lookup = AffineRowLRU.lookup
        original_reader = SafeTensorMMap.affine2_rows

        def outputs(engine_core_outputs, *args, **kwargs):
            result = original_outputs(engine_core_outputs, *args, **kwargs)
            mark = time.perf_counter_ns()
            request = self.active
            if request is not None:
                finished = {out.request_id for out in result.request_outputs if out.finished}
                for out in engine_core_outputs:
                    if out.new_token_ids:
                        request["token_batches"] += 1
                        request["observed_tokens"] += len(out.new_token_ids)
                        request["last_nonempty_observed_ns"] = time.monotonic_ns()
                        if request["first_token"] is None:
                            request["first_token"] = self._snapshot()
                        if out.finished or out.request_id in finished:
                            request["last_token"] = (
                                request["first_token"] if request["token_batches"] == 1
                                else self._snapshot()
                            )
            self.overhead_ns += time.perf_counter_ns() - mark
            return result

        def lookup(cache, ids):
            entered = time.perf_counter_ns()
            if cache not in self._caches:
                self._caches[cache] = set()
                if self.active is not None:
                    key = str(id(cache))
                    self.active["start"]["engram_cache"][key] = self._cache_stats()[key]
            self._caches[cache].add(threading.get_native_id())
            self.overhead_ns += time.perf_counter_ns() - entered
            return original_lookup(cache, ids)

        def reader(source, base, rows):
            request = self.active
            if request is None:
                return original_reader(source, base, rows)
            entered = time.perf_counter_ns()
            if not hasattr(rows, "__len__"):
                rows = list(rows)
            phase = "before_first" if request["first_token"] is None else "decode"
            key = (str(source.path), base, phase, threading.get_native_id())
            before = _usage(resource.RUSAGE_THREAD)
            started = time.perf_counter_ns()
            try:
                return original_reader(source, base, rows)
            finally:
                ended = time.perf_counter_ns()
                after = _usage(resource.RUSAGE_THREAD)
                if key not in request["reader"]:
                    ranges = {}
                    for suffix in ("weight", "scales", "biases"):
                        name = base + "." + suffix
                        entry = source.entries.get(name)
                        if entry is not None:
                            ranges[name] = {
                                "tensor_offset_start": entry.offset,
                                "tensor_offset_end_exclusive": entry.offset + entry.nbytes,
                                "row_bytes": entry.nbytes // entry.shape[0],
                            }
                    request["reader"][key] = {
                        "path": key[0], "base": base, "phase": phase, "tid": key[3],
                        "calls": 0, "requested_rows": 0, "logical_source_bytes": 0,
                        "wall_ns": 0, "rusage_thread": dict.fromkeys(_RU, 0),
                        "tensor_ranges": ranges,
                    }
                bucket = request["reader"][key]
                bucket["calls"] += 1
                bucket["requested_rows"] += len(rows)
                bucket["logical_source_bytes"] += len(rows) * sum(
                    item["row_bytes"] for item in bucket["tensor_ranges"].values())
                bucket["wall_ns"] += ended - started
                for field, value in _delta(after, before).items():
                    bucket["rusage_thread"][field] += value
                self.overhead_ns += time.perf_counter_ns() - entered - (ended - started)

        self._replace(self.processor, "process_outputs", outputs)
        self._replace(AffineRowLRU, "lookup", lookup)
        self._replace(SafeTensorMMap, "affine2_rows", reader)
        return self

    def begin_request(self, label):
        if self.active is not None or not self._patches:
            raise RuntimeError("install diagnostics and end the previous request first")
        entered = time.perf_counter_ns()
        self.active = {
            "label": label, "start": self._snapshot(), "first_token": None,
            "last_token": None, "last_nonempty_observed_ns": None,
            "observed_tokens": 0, "token_batches": 0, "reader": {},
            "overhead_start_ns": self.overhead_ns,
        }
        self.overhead_ns += time.perf_counter_ns() - entered

    def end_request(self):
        if self.active is None:
            raise RuntimeError("no active diagnostic request")
        entered = time.perf_counter_ns()
        result = self.active
        result["end"] = self._snapshot()
        self.active = None
        result["reader"] = list(result["reader"].values())
        result["intervals"] = {}
        for name, start, end in (
            ("request_to_first", result["start"], result["first_token"]),
            ("first_to_last", result["first_token"], result["last_token"]),
        ):
            if start is not None and end is not None:
                result["intervals"][name] = {
                    "wall_ns": end["monotonic_ns"] - start["monotonic_ns"],
                    **{field: _delta(end[field], start[field]) for field in
                       ("rusage_self", "rusage_children", "process_io", "global_vmstat")},
                    "engram_cache_delta": {
                        key: {field: cache[field] - start["engram_cache"].get(key, {}).get(field, 0)
                              for field in ("hits", "misses", "rows_read")}
                        for key, cache in end["engram_cache"].items()
                    },
                }
        result["last_token_snapshot_available"] = result["last_token"] is not None
        result["caveats"] = [
            "Logical source bytes are not physical disk bytes; reader wall includes CPU work.",
            "Reader thread faults are interval counts, without fault-address attribution.",
            "Global vmstat includes other processes; RUSAGE_CHILDREN covers reaped children only.",
            "Live child work is unmeasured; output observations include output processing.",
            "No last-token resource snapshot if termination arrives in a later empty batch.",
            "Instrumentation is included in measurements; do not subtract it from throughput.",
        ]
        self.overhead_ns += time.perf_counter_ns() - entered
        result["instrumentation_overhead_ns"] = self.overhead_ns - result.pop("overhead_start_ns")
        return result

    def snapshot_maps(self, label):
        if self.active is not None:
            raise RuntimeError("maps snapshots must be outside timed requests")
        entered = time.perf_counter_ns()
        errors = {}
        maps = _read("/proc/self/maps", errors)
        regions = []
        for line in _read("/proc/self/smaps", errors).splitlines():
            if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                fields = line.split(None, 5)
                region = dict(zip(("address", "permissions", "offset", "device", "inode"), fields))
                region["path"] = fields[5] if len(fields) > 5 else "[anonymous]"
                regions.append(region)
            elif regions and line.split(":", 1)[0] in {"Rss", "Swap", "Anonymous", "Private_Dirty"}:
                regions[-1].update(_numbers(line))
        grouped = defaultdict(lambda: {"mappings": 0, "Rss": 0, "Swap": 0, "Anonymous": 0})
        for region in regions:
            group = grouped[region["path"]]
            group["mappings"] += 1
            for key in ("Rss", "Swap", "Anonymous"):
                group[key] += region.get(key, 0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(label))
        path = self.output_dir / f"fault_maps_{os.getpid()}_{len(self.map_snapshots)}_{safe_label}.json"
        payload = {
            "label": label, "pid": os.getpid(), "rank": self.rank,
            "monotonic_ns": time.monotonic_ns(), "maps": maps, "regions": regions,
            "by_path_kib": dict(grouped), "read_errors": errors,
            "read_parse_overhead_ns": time.perf_counter_ns() - entered,
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        result = {"label": label, "path": str(path),
                  "total_overhead_ns": time.perf_counter_ns() - entered}
        self.map_snapshots.append(result)
        return result

    def close(self):
        for target, name, original, had_own, replacement in reversed(self._patches):
            if getattr(target, name) is replacement:
                if had_own:
                    setattr(target, name, original)
                else:
                    delattr(target, name)
        self._patches.clear()
        self.active = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
