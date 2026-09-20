#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
RAW = Path('/home/funboy/reports/DS4-SPEED-001')
COLLECTOR = ROOT / 'scripts/run-ds4-usable-request.py'
MANIFEST = json.loads((ROOT / 'runtime/ds41/document-profile-002/prompt-manifest.json').read_text())


def atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def record(ident: str) -> dict:
    for group in ('documents', 'holdouts'):
        for item in MANIFEST[group]:
            if item['id'] == ident:
                return item
    raise KeyError(ident)


def run_case(out: Path, tag: str, ident: str, index: int, diagnostic: bool) -> dict:
    spec = record(ident)
    case_id = f'{tag}-{ident}-{index}'
    log = RAW / 'runtime' / tag / 'coordinator.log'
    command = [
        'python3', str(COLLECTOR), '--id', case_id, '--out-dir', str(out),
        '--prompt-file', str(ROOT / spec['prompt_file']), '--max-tokens', '2048',
        '--thinking', 'on', '--reasoning-effort', 'low', '--server-log', str(log),
        '--timeout', '1800',
    ]
    env = os.environ.copy()
    env['DS4_COORDINATOR_UNIT'] = 'ds4-speed-001-coordinator.service'
    env['DS4_WORKER_UNIT'] = 'ds4-speed-001-worker.service'
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    (out / f'{case_id}.collector.stdout').write_text(completed.stdout)
    (out / f'{case_id}.collector.stderr').write_text(completed.stderr)
    if completed.returncode:
        return {'id': case_id, 'status': 'FAILED_TRANSPORT', 'rc': completed.returncode,
                'stderr': completed.stderr[-4000:], 'diagnostic': diagnostic}
    result = json.loads((out / case_id / 'result.json').read_text())
    state = result['state']
    content = state.get('content') or ''
    try:
        actual = json.loads(content)
        semantic = actual == spec['expected']
    except Exception:
        actual = None
        semantic = False
    server = result.get('server') or {}
    prefill = server.get('prefill') or {}
    decode = server.get('decode') or {}
    cache = result.get('cache') or {}
    row = {
        'id': case_id,
        'case': ident,
        'diagnostic': diagnostic,
        'status': 'PASS' if semantic and state.get('finish_reason') == 'stop' else 'FAIL',
        'semantic_pass': semantic,
        'expected': spec['expected'],
        'actual': actual,
        'finish_reason': state.get('finish_reason'),
        'prompt_tokens': cache.get('prompt_tokens'),
        'completion_tokens': cache.get('completion_tokens'),
        'cached_tokens': cache.get('cached_tokens'),
        'prefill_tps': prefill.get('avg_tps'),
        'prefill_s': prefill.get('engine_s'),
        'decode_tps': decode.get('avg_tps'),
        'decode_s': decode.get('engine_decode_s'),
        'first_final_s': state.get('first_final_s'),
        'wall_s': result.get('wall_s'),
        'reasoning_chars': len(state.get('reasoning') or ''),
        'final_chars': len(content),
        'resources_before': result.get('resources_before'),
        'resources_after': result.get('resources_after'),
    }
    atomic(out / case_id / 'validation.json', row)
    return row


def median(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get('status') == 'PASS' and isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--arm', required=True, choices=('serial', 'concurrent'))
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--cases', help='comma-separated manifest case ids')
    args = parser.parse_args()
    out = RAW / 'ab' / args.tag
    if out.exists():
        raise SystemExit(f'replay guard: {out}')
    out.mkdir(parents=True)
    status = json.loads(subprocess.check_output([str(ROOT / 'scripts/ds4-speed-001-controller.sh'), 'status'], text=True))
    if status.get('state') != 'READY':
        raise SystemExit(f'pair not ready: {status}')
    rows = []
    cases = tuple(x.strip() for x in args.cases.split(',') if x.strip()) if args.cases else (
        ('code2k-middle-explicit-v2',) if args.diagnostic else
        ('code2k-middle-explicit-v2', 'docs2k-middle-explicit-v2'))
    if not cases:
        raise SystemExit('at least one case is required')
    samples = 1 if args.diagnostic else args.samples
    for index in range(1, samples + 1):
        for ident in cases:
            row = run_case(out, args.tag, ident, index, args.diagnostic)
            rows.append(row)
            atomic(out / 'registry.json', {
                'schema': 'ds4-speed-001-arm-v1', 'state': 'IN_FLIGHT',
                'arm': args.arm, 'tag': args.tag, 'rows': rows, 'updated_unix': time.time(),
            })
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row['status'] != 'PASS':
                break
        if rows[-1]['status'] != 'PASS':
            break
    terminal = {
        'schema': 'ds4-speed-001-arm-v1',
        'state': 'COMPLETE',
        'arm': args.arm,
        'tag': args.tag,
        'diagnostic': args.diagnostic,
        'status': 'PASS' if len(rows) == samples * len(cases) and all(r['status'] == 'PASS' for r in rows) else 'FAIL',
        'rows': rows,
        'medians': {key: median(rows, key) for key in ('prefill_tps', 'decode_tps', 'first_final_s', 'wall_s')},
        'finished_unix': time.time(),
    }
    atomic(out / 'terminal.json', terminal)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
