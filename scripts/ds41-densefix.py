#!/usr/bin/env python3
"""Selective repair tool for DS41 MixedQ2 FP8-derived BF16 dense tensors.

This tool NEVER edits the original MixedQ2 artifact.  It builds a manifest from
an immutable 328-tensor map, downloads only exact byte ranges from the pinned
DeepSeek V4.1 source revision, creates a physically independent copy of the
five GGUF shards, and patches only the listed BF16 payload ranges.

Converter reference (intentionally independent from validator):
  * E4M3FN is decoded by an explicit NumPy lookup table.
  * E8M0 is decoded from raw biased exponent bits.
  * BF16 uses explicit round-to-nearest-even on IEEE float32 bits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / ".vendor" / "llama-v41" / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

SPEC_DEFAULT = ROOT / "runtime" / "ds41" / "densefix-v1-map.json"
REPORT_DEFAULT = ROOT / "reports" / "DS41-Q2-001" / "densefix"
ORIGINAL_DEFAULT = Path("/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2")
DEST_DEFAULT = Path("/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix")
CACHE_DEFAULT = Path("/home/funboy/models/ds41/densefix-source-cache")
CHUNK = 8 * 1024 * 1024
COPY_CHUNK = 64 * 1024 * 1024


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "DS41-Q2-001-densefix/1", "Accept-Encoding": "identity"})
    return s


def range_url(repo: str, rev: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{filename}"


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", value or "")
    if not m or m.group(3) == "*":
        raise RuntimeError(f"invalid Content-Range: {value!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def strict_get_range(sess: requests.Session, url: str, start: int, end: int, *, stream: bool = True):
    if start < 0 or end < start:
        raise ValueError((start, end))
    r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=stream, timeout=(15, 120))
    if r.status_code != 206:
        r.close()
        raise RuntimeError(f"Range request rejected: HTTP {r.status_code} for {url} bytes={start}-{end}")
    a, b, total = parse_content_range(r.headers.get("Content-Range"))
    if (a, b) != (start, end):
        r.close()
        raise RuntimeError(f"wrong Content-Range {(a,b,total)} expected {(start,end)}")
    clen = int(r.headers.get("Content-Length", "-1"))
    if clen != end - start + 1:
        r.close()
        raise RuntimeError(f"wrong Content-Length {clen} expected {end-start+1}")
    return r, total


def fetch_header(sess: requests.Session, repo: str, rev: str, shard: str, header_dir: Path) -> dict:
    header_dir.mkdir(parents=True, exist_ok=True)
    raw_path = header_dir / f"{shard}.header.raw.json"
    meta_path = header_dir / f"{shard}.header.meta.json"
    url = range_url(repo, rev, shard)
    r, total = strict_get_range(sess, url, 0, 7)
    first8 = r.content
    r.close()
    if len(first8) != 8:
        raise RuntimeError(f"short safetensors header length for {shard}")
    header_len = struct.unpack("<Q", first8)[0]
    r, total2 = strict_get_range(sess, url, 8, 8 + header_len - 1)
    raw = r.content
    r.close()
    if total2 != total or len(raw) != header_len:
        raise RuntimeError(f"header length/total changed for {shard}")
    header = json.loads(raw)
    raw_path.write_bytes(raw)
    meta = {
        "repo": repo,
        "revision": rev,
        "shard": shard,
        "file_size": total,
        "header_len": header_len,
        "data_base": 8 + header_len,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "first8_hex": first8.hex(),
    }
    atomic_json(meta_path, meta)
    return {"header": header, "meta": meta}


def original_sha_map(original: Path) -> dict[str, str]:
    p = original / "SHA256SUMS"
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[-1].lstrip("*")] = parts[0]
    return out


def build_manifest(args) -> None:
    from gguf import GGUFReader, GGMLQuantizationType

    spec = json.loads(Path(args.spec).read_text())
    entries = spec["entries"]
    original = Path(args.original)
    report = Path(args.report)
    header_dir = report / "source-headers-strict"
    report.mkdir(parents=True, exist_ok=True)

    # Locate every selected tensor in exactly one original GGUF shard.
    located: dict[str, dict] = {}
    shard_meta: dict[str, dict] = {}
    orig_sha = original_sha_map(original)
    for gp in sorted(original.glob("*.gguf")):
        reader = GGUFReader(str(gp), "r")
        wanted = {e["gguf"] for e in entries}
        intervals = []
        for t in reader.tensors:
            if t.name not in wanted:
                continue
            if t.name in located:
                raise RuntimeError(f"duplicate selected GGUF tensor {t.name}")
            if t.tensor_type != GGMLQuantizationType.BF16:
                raise RuntimeError(f"selected tensor is not BF16: {t.name} {t.tensor_type}")
            located[t.name] = {
                "gguf_shard": gp.name,
                "gguf_offset": int(t.data_offset),
                "gguf_nbytes": int(t.n_bytes),
                "gguf_shape_header": [int(x) for x in t.shape.tolist()],
            }
            intervals.append((int(t.data_offset), int(t.data_offset + t.n_bytes), t.name))
        intervals.sort()
        for a, b in zip(intervals, intervals[1:]):
            if a[1] > b[0]:
                raise RuntimeError(f"overlapping GGUF patch intervals in {gp.name}: {a} {b}")
        shard_meta[gp.name] = {
            "size": gp.stat().st_size,
            "original_sha256_expected": orig_sha.get(gp.name),
            "patch_intervals": [{"start": a, "end": b, "tensor": n} for a, b, n in intervals],
        }
    if len(located) != len(entries):
        missing = sorted({e["gguf"] for e in entries} - set(located))
        raise RuntimeError(f"missing selected GGUF tensors: {missing[:20]} count={len(missing)}")

    sess = session()
    headers: dict[str, dict] = {}
    for shard in sorted({e["source_shard"] for e in entries}):
        headers[shard] = fetch_header(sess, spec["source_repo"], spec["source_revision"], shard, header_dir)
        print(f"MANIFEST_HEADER {shard}", flush=True)

    out_entries = []
    sum_weight = sum_scale = sum_gguf = 0
    for idx, e in enumerate(entries):
        h = headers[e["source_shard"]]
        hdr = h["header"]
        wm = hdr.get(e["source"])
        sm = hdr.get(e["scale"])
        if wm is None or sm is None:
            raise RuntimeError(f"source/scale missing for {e['source']} in {e['source_shard']}")
        if wm["dtype"] != "F8_E4M3" or sm["dtype"] != "F8_E8M0":
            raise RuntimeError(f"unexpected source dtypes {e['source']}: {wm['dtype']} {sm['dtype']}")
        shape = [int(x) for x in wm["shape"]]
        if shape != [int(x) for x in e["logical_shape"]]:
            raise RuntimeError(f"source/GGUF shape mismatch {e['source']}: {shape} vs {e['logical_shape']}")
        m, k = shape
        expected_scale_shape = [math.ceil(m / 32), math.ceil(k / 32)]
        if [int(x) for x in sm["shape"]] != expected_scale_shape:
            raise RuntimeError(f"bad 32x32 scale shape {e['scale']}: {sm['shape']} expected {expected_scale_shape}")
        data_base = int(h["meta"]["data_base"])
        w0, w1 = map(int, wm["data_offsets"])
        s0, s1 = map(int, sm["data_offsets"])
        wlen, slen = w1 - w0, s1 - s0
        if wlen != m * k or slen != expected_scale_shape[0] * expected_scale_shape[1]:
            raise RuntimeError(f"source byte lengths invalid for {e['source']}")
        loc = located[e["gguf"]]
        if loc["gguf_nbytes"] != m * k * 2 or loc["gguf_nbytes"] != int(e["gguf_bytes"]):
            raise RuntimeError(f"GGUF BF16 bytes invalid for {e['gguf']}")
        key = hashlib.sha256(e["source"].encode()).hexdigest()[:16]
        row = {
            "index": idx,
            **e,
            **loc,
            "source_dtype": wm["dtype"],
            "scale_dtype": sm["dtype"],
            "source_shape": shape,
            "scale_shape": expected_scale_shape,
            "source_file_size": int(h["meta"]["file_size"]),
            "source_header_len": int(h["meta"]["header_len"]),
            "source_header_sha256": h["meta"]["header_sha256"],
            "weight_range": {"start": data_base + w0, "end": data_base + w1, "nbytes": wlen},
            "scale_range": {"start": data_base + s0, "end": data_base + s1, "nbytes": slen},
            "cache_weight_file": f"tensors/{idx:03d}-{key}.weight.bin",
            "cache_scale_file": f"tensors/{idx:03d}-{key}.scale.bin",
        }
        out_entries.append(row)
        sum_weight += wlen
        sum_scale += slen
        sum_gguf += loc["gguf_nbytes"]

    manifest = {
        "schema": "ds41-densefix-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_commit": args.tool_commit,
        "tool_path": "scripts/ds41-densefix.py",
        "validator_path": "scripts/validate-ds41-densefix.py",
        "original_path": str(original),
        "derived_path": str(Path(args.dest)),
        "cache_path": str(Path(args.cache)),
        "original_repo": spec["original_repo"],
        "original_revision": spec["original_revision"],
        "source_repo": spec["source_repo"],
        "source_revision": spec["source_revision"],
        "conversion": "F8_E4M3 numerical decode * F8_E8M0 32x32 numerical scale -> float32 -> BF16 RNE",
        "counts": {"tensors": len(out_entries), "source_shards": len(headers)},
        "bytes": {"source_weight": sum_weight, "source_scale": sum_scale, "source_total": sum_weight + sum_scale, "gguf_repaired": sum_gguf},
        "gguf_shards": shard_meta,
        "entries": out_entries,
    }
    if len(out_entries) != 328 or sum_weight + sum_scale != 6528512000:
        raise RuntimeError(f"inventory drift count={len(out_entries)} source_bytes={sum_weight+sum_scale}")
    atomic_json(Path(args.manifest), manifest)
    print(json.dumps({"status": "PASS", "manifest": args.manifest, "counts": manifest["counts"], "bytes": manifest["bytes"]}, indent=2))


def download_one(sess: requests.Session, url: str, start: int, end_exclusive: int, path: Path, max_attempts: int = 12) -> dict:
    expected = end_exclusive - start
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > expected:
        raise RuntimeError(f"oversized partial range file {path}")
    attempts = 0
    while path.stat().st_size if path.exists() else 0 < expected:
        cur = path.stat().st_size if path.exists() else 0
        if cur == expected:
            break
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"too many resume attempts for {path} at {cur}/{expected}")
        req_start = start + cur
        try:
            r, total = strict_get_range(sess, url, req_start, end_exclusive - 1)
            mode = "ab" if cur else "wb"
            wrote = 0
            with path.open(mode, buffering=0) as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        wrote += len(chunk)
                f.flush(); os.fsync(f.fileno())
            r.close()
            if wrote <= 0:
                raise RuntimeError("range response wrote zero bytes")
        except Exception:
            time.sleep(min(10, attempts))
            if attempts >= max_attempts:
                raise
    if path.stat().st_size != expected:
        raise RuntimeError(f"range file size mismatch {path}: {path.stat().st_size} != {expected}")
    return {"nbytes": expected, "sha256": sha256_file(path), "attempts": attempts}


def download(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    cache = Path(args.cache)
    state_path = cache / "download-receipts.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"schema": "ds41-densefix-download-v1", "completed": {}}
    sess = session()
    total = len(manifest["entries"])
    for i, e in enumerate(manifest["entries"], 1):
        url = range_url(manifest["source_repo"], manifest["source_revision"], e["source_shard"])
        key = str(e["index"])
        rec = state["completed"].get(key, {})
        for kind, rk, fk in (("weight", "weight_range", "cache_weight_file"), ("scale", "scale_range", "cache_scale_file")):
            rr = e[rk]
            p = cache / e[fk]
            got = download_one(sess, url, int(rr["start"]), int(rr["end"]), p)
            rec[kind] = {**got, "file": e[fk], "range": [int(rr["start"]), int(rr["end"])], "source_shard": e["source_shard"]}
        state["completed"][key] = rec
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(state_path, state)
        if i == 1 or i % 8 == 0 or i == total:
            done_bytes = sum(v2.get(k, {}).get("nbytes", 0) for v2 in state["completed"].values() for k in ("weight", "scale"))
            print(f"DOWNLOAD_PROGRESS tensors={i}/{total} bytes={done_bytes}/{manifest['bytes']['source_total']}", flush=True)
    if len(state["completed"]) != total:
        raise RuntimeError("download state incomplete")
    print(json.dumps({"status":"PASS","tensors":total,"bytes":manifest["bytes"]["source_total"],"receipt":str(state_path)}, indent=2))


def intervals_for_shard(manifest: dict, shard: str) -> list[tuple[int,int]]:
    xs = [(int(x["start"]), int(x["end"])) for x in manifest["gguf_shards"][shard]["patch_intervals"]]
    xs.sort()
    return xs


def update_unpatched_hash(h, chunk: bytes, base: int, intervals: list[tuple[int,int]], idx: int) -> int:
    end = base + len(chunk)
    while idx < len(intervals) and intervals[idx][1] <= base:
        idx += 1
    cursor = base
    j = idx
    while j < len(intervals) and intervals[j][0] < end:
        a, b = intervals[j]
        if cursor < min(a, end):
            h.update(chunk[cursor-base:min(a,end)-base])
        cursor = max(cursor, min(b, end))
        if b <= end:
            j += 1
        else:
            break
    if cursor < end:
        h.update(chunk[cursor-base:])
    return idx


def prepare(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    original = Path(args.original); dest = Path(args.dest)
    if dest.exists():
        raise RuntimeError(f"destination already exists; refusing ambiguous overwrite: {dest}")
    dest.mkdir(parents=True)
    state = {"schema":"ds41-densefix-precopy-v1","original":str(original),"dest":str(dest),"shards":{},"tool_commit":manifest["tool_commit"]}
    for shard in sorted(manifest["gguf_shards"]):
        src = original / shard; dst = dest / shard
        intervals = intervals_for_shard(manifest, shard)
        full = hashlib.sha256(); unchanged = hashlib.sha256(); idx = 0; pos = 0
        with src.open("rb", buffering=0) as fi, dst.open("xb", buffering=0) as fo:
            while True:
                b = fi.read(COPY_CHUNK)
                if not b: break
                full.update(b)
                idx = update_unpatched_hash(unchanged, b, pos, intervals, idx)
                fo.write(b); pos += len(b)
            fo.flush(); os.fsync(fo.fileno())
        os.chmod(dst, stat.S_IMODE(src.stat().st_mode))
        if src.stat().st_ino == dst.stat().st_ino or dst.stat().st_nlink != 1:
            raise RuntimeError(f"destination shard is not physically independent: {dst}")
        got = full.hexdigest(); expected = manifest["gguf_shards"][shard].get("original_sha256_expected")
        if expected and got != expected:
            raise RuntimeError(f"original SHA mismatch while copying {shard}: {got} != {expected}")
        state["shards"][shard] = {"size":pos,"original_sha256":got,"unpatched_sha256":unchanged.hexdigest(),"dest_inode":dst.stat().st_ino,"dest_nlink":dst.stat().st_nlink}
        atomic_json(dest / ".densefix-precopy.json", state)
        print(f"COPY_COMPLETE {shard} bytes={pos} sha256={got}", flush=True)
    for p in original.iterdir():
        if p.name.endswith(".gguf"): continue
        target = dest / p.name
        if p.is_dir(): shutil.copytree(p, target, symlinks=True)
        elif p.is_file(): shutil.copy2(p, target)
    # Keep original SHA file as provenance; repaired hashes are written separately by validator.
    atomic_json(dest / "densefix-manifest.json", manifest)
    print(json.dumps({"status":"PASS","dest":str(dest),"shards":state["shards"]}, indent=2))


def fp8_e4m3fn_table() -> np.ndarray:
    out = np.empty(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0:
            v = mant * (2.0 ** -9)
        elif exp == 0xF and mant == 0x7:
            v = math.nan
        else:
            v = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
        out[b] = np.float32(sign * v)
    return out


def e8m0_table() -> np.ndarray:
    u = (np.arange(256, dtype=np.uint32) << np.uint32(23)).astype(np.uint32)
    return u.view(np.float32)


def f32_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite value during dense dequantization")
    bits = x.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2")


def selftest() -> None:
    ft = fp8_e4m3fn_table(); st = e8m0_table()
    x = np.float32(ft[0x71] * st[0x73])
    bits = int(f32_to_bf16_rne(np.array([x],dtype=np.float32))[0])
    assert float(ft[0x71]) == 144.0
    assert float(st[0x73]) == 2.0 ** -12
    assert float(x) == 0.03515625 and bits == 0x3D10, (x,hex(bits))
    # Known float8_e4m3fn extrema/specials.
    assert float(ft[0x7E]) == 448.0 and float(ft[0xFE]) == -448.0 and math.isnan(float(ft[0x7F]))
    print(json.dumps({"status":"PASS","fp8_0x71":float(ft[0x71]),"e8m0_0x73":float(st[0x73]),"product":float(x),"bf16_bits":hex(bits)}))


def repair(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    cache = Path(args.cache); dest = Path(args.dest)
    if not (dest / ".densefix-precopy.json").exists():
        raise RuntimeError("derived artifact lacks successful precopy receipt")
    dl = json.loads((cache / "download-receipts.json").read_text())
    if len(dl.get("completed",{})) != len(manifest["entries"]):
        raise RuntimeError("source download receipt incomplete")
    ft = fp8_e4m3fn_table(); st = e8m0_table()
    if np.isnan(ft).any():
        # NaN codes are allowed in the table but not in actual weights; checked per block.
        pass
    receipts = {"schema":"ds41-densefix-repair-v1","tool_commit":manifest["tool_commit"],"completed":{}}
    out_state = dest / ".densefix-repair.json"
    for i, e in enumerate(manifest["entries"], 1):
        wpath = cache / e["cache_weight_file"]; spath = cache / e["cache_scale_file"]
        m, k = map(int, e["source_shape"]); sm, sk = map(int, e["scale_shape"])
        w = np.memmap(wpath, mode="r", dtype=np.uint8, shape=(m,k))
        s = np.memmap(spath, mode="r", dtype=np.uint8, shape=(sm,sk))
        out_hash = hashlib.sha256(); nonfinite = 0
        dst_path = dest / e["gguf_shard"]
        fd = os.open(dst_path, os.O_RDWR)
        try:
            for r0 in range(0, m, 32):
                r1 = min(m, r0 + 32)
                wb = np.asarray(w[r0:r1])
                vals = ft[wb]
                sb = np.asarray(s[r0 // 32])
                scales = np.repeat(st[sb], 32)[:k]
                f = vals * scales[np.newaxis, :]
                nonfinite += int((~np.isfinite(f)).sum())
                if nonfinite:
                    raise RuntimeError(f"nonfinite source dequant {e['source']}")
                bf = f32_to_bf16_rne(f)
                blob = bf.tobytes(order="C")
                off = int(e["gguf_offset"]) + r0 * k * 2
                n = os.pwrite(fd, blob, off)
                if n != len(blob):
                    raise RuntimeError(f"short pwrite {e['gguf']} {n}/{len(blob)}")
                out_hash.update(blob)
            os.fsync(fd)
        finally:
            os.close(fd)
        rec = dl["completed"][str(e["index"])]
        receipts["completed"][str(e["index"])] = {
            "gguf": e["gguf"], "gguf_shard": e["gguf_shard"], "gguf_offset": e["gguf_offset"], "gguf_nbytes": e["gguf_nbytes"],
            "source": e["source"], "scale": e["scale"],
            "source_weight_sha256": rec["weight"]["sha256"], "source_scale_sha256": rec["scale"]["sha256"],
            "repaired_payload_sha256": out_hash.hexdigest(), "rows": m, "cols": k,
        }
        atomic_json(out_state, receipts)
        if i == 1 or i % 8 == 0 or i == len(manifest["entries"]):
            print(f"REPAIR_PROGRESS tensors={i}/{len(manifest['entries'])}", flush=True)
    if len(receipts["completed"]) != 328:
        raise RuntimeError("repair receipt incomplete")
    print(json.dumps({"status":"PASS","tensors":328,"receipt":str(out_state)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("selftest","manifest","download","prepare","repair"))
    p.add_argument("--spec", default=str(SPEC_DEFAULT))
    p.add_argument("--manifest", default=str(REPORT_DEFAULT / "densefix-manifest.plan.json"))
    p.add_argument("--report", default=str(REPORT_DEFAULT))
    p.add_argument("--original", default=str(ORIGINAL_DEFAULT))
    p.add_argument("--dest", default=str(DEST_DEFAULT))
    p.add_argument("--cache", default=str(CACHE_DEFAULT))
    p.add_argument("--tool-commit", default="UNCOMMITTED")
    args = p.parse_args()
    if args.command == "selftest": selftest()
    elif args.command == "manifest": build_manifest(args)
    elif args.command == "download": download(args)
    elif args.command == "prepare": prepare(args)
    elif args.command == "repair": repair(args)

if __name__ == "__main__":
    main()
