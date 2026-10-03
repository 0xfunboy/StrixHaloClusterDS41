#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path("/home/funboy/StrixHaloClusterDS41")
P3 = ROOT / "runtime/ds41/long-context-baseline-001/p3-frozen-manifest.json"
OUT = ROOT / "runtime/ds41/long-context-baseline-001"
ADDED = ROOT / "QUALIFICATION.md"
LABELS = ("4k","8k","16k","32k")

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def strip_task(text: str) -> str:
    i = text.rfind("\nTASK:")
    if i < 0:
        raise RuntimeError("TASK marker missing")
    return text[:i].rstrip() + "\n"

def section_names(text: str) -> list[str]:
    rows = re.findall(r"^===== FILE (.+?) SHA256 [0-9a-f]{64} =====$", text, re.M)
    return [Path(x).name for x in rows]

def first_h1(text: str) -> str:
    m = re.search(r"^# (.+?)\s*$", text, re.M)
    if not m:
        raise RuntimeError("added file H1 missing")
    return m.group(1)

def first_h2_year(text: str) -> int:
    m = re.search(r"^## .*?((?:19|20)\d{2}).*$", text, re.M)
    if not m:
        raise RuntimeError("added file H2 year missing")
    return int(m.group(1))

p3 = json.loads(P3.read_text())
by_id = {x["id"]: x for x in p3["originals"]}
added_text = ADDED.read_text()
added_bytes = ADDED.read_bytes()
rows = []

for label in LABELS:
    src = by_id[f"docs-{label}"]
    prompt_path = ROOT / src["prompt_file"]
    base = strip_task(prompt_path.read_text())
    names = section_names(base)
    if len(names) < 4:
        raise RuntimeError(f"{label}: too few sections")
    f = src["facts"]

    q1 = (
        "\nP4 QUESTION 1: Using only the corpus above, return exactly one JSON object "
        "with keys result, first_file, last_file. result must equal "
        "(DISTANT_FACT_BEGIN + DISTANT_FACT_END). first_file and last_file are "
        "the corresponding basenames by appearance order. No prose.\n"
    )
    q2 = (
        "P4 QUESTION 2: Continue from the same corpus. Return exactly one JSON object "
        "with keys result, middle_file. result must equal "
        "(DISTANT_FACT_BEGIN + DISTANT_FACT_MIDDLE - DISTANT_FACT_END). "
        "middle_file is the lower central FILE section: if N is even use section N/2, "
        "otherwise section (N+1)/2. No prose."
    )
    added_block = (
        "\n===== ADDED FILE QUALIFICATION.md =====\n" + added_text.rstrip() +
        "\n===== END ADDED FILE =====\n\n"
    )
    q3 = (
        added_block +
        "P4 QUESTION 3: Use both the original corpus and the added file. Return exactly "
        "one JSON object with keys result, added_title, added_year. result must equal "
        "(DISTANT_FACT_BEGIN + DISTANT_FACT_MIDDLE + DISTANT_FACT_END). added_title "
        "is the text of the first level-1 Markdown heading in the added file, without "
        "the leading #. added_year is the four-digit year in the first level-2 heading "
        "of the added file. No prose."
    )
    q4 = (
        "P4 QUESTION 4: Resume the original conversation and corpus. Return exactly one "
        "JSON object with keys result, penultimate_file. result must equal "
        "(DISTANT_FACT_BEGIN * DISTANT_FACT_MIDDLE - DISTANT_FACT_END). "
        "penultimate_file is the basename of the penultimate FILE section in the "
        "original corpus. No prose."
    )

    n = len(names)
    mid = (n - 1) // 2
    row = {
        "label": label,
        "base_source_id": src["id"],
        "base_prompt_file": src["prompt_file"],
        "base_prompt_sha256": src["prompt_sha256"],
        "base_rendered_low_tokens_p3": src["rendered_low_tokens"],
        "facts": f,
        "section_count": n,
        "added_file": "QUALIFICATION.md",
        "added_file_sha256": sha(added_bytes),
        "added_file_bytes": len(added_bytes),
        "q1": q1,
        "q1_sha256": sha(q1.encode()),
        "expected_q1": {
            "result": f["begin"] + f["end"],
            "first_file": names[0],
            "last_file": names[-1],
        },
        "q2": q2,
        "q2_sha256": sha(q2.encode()),
        "expected_q2": {
            "result": f["begin"] + f["middle"] - f["end"],
            "middle_file": names[mid],
        },
        "q3": q3,
        "q3_sha256": sha(q3.encode()),
        "expected_q3": {
            "result": f["begin"] + f["middle"] + f["end"],
            "added_title": first_h1(added_text),
            "added_year": first_h2_year(added_text),
        },
        "q4": q4,
        "q4_sha256": sha(q4.encode()),
        "expected_q4": {
            "result": f["begin"] * f["middle"] - f["end"],
            "penultimate_file": names[-2],
        },
        "isolation_prompt": "Return exactly ISOLATION-OK and nothing else.",
        "isolation_expected": "ISOLATION-OK",
    }
    base_file = OUT / "p4" / f"docs-{label}-base.txt"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    base_file.write_text(base)
    row["base_file"] = str(base_file.relative_to(ROOT))
    row["base_file_sha256"] = sha(base.encode())
    rows.append(row)

manifest = {
    "schema": "ds4-long-context-baseline-001-p4-frozen-v1",
    "frozen_before_inference": True,
    "profile": {
        "reasoning_effort": "low",
        "temperature": 0,
        "seed": 1,
        "output_cap": 2048,
        "context_window": 65536,
    },
    "selection_rule": "highest P3 level <=32k with both code and docs originals PASS; then verify each rendered P4 request fits prompt+2048<=65536",
    "candidates": rows,
    "notes": [
        "Questions and validators are frozen before P4 requests.",
        "Actual assistant final contents are inserted verbatim into subsequent history.",
        "Step 4 reuses the exact step-3 message list but runs after a whole-pair fresh restart.",
        "Step 5 is an independent isolation control; step 6 resumes the original full history."
    ],
}
atomic(OUT / "p4-frozen-manifest.json", manifest)
print(json.dumps({
    "status": "PASS",
    "candidates": [x["label"] for x in rows],
    "added_sha256": sha(added_bytes),
    "added_bytes": len(added_bytes),
}, indent=2))
