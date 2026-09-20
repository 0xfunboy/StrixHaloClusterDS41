#!/usr/bin/env python3
from __future__ import annotations

import http.client
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
OUT = Path('/home/funboy/reports/DS4-SPEED-001/soak-e1')
STATE = Path('/home/funboy/.local/state/ds4-document-profile-002')
MANIFEST = json.loads((ROOT / 'runtime/ds41/document-profile-002/prompt-manifest.json').read_text())


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def identity() -> dict:
    local = subprocess.run(
        ['systemctl', '--user', 'show', 'ds4-speed-001-coordinator.service',
         '-p', 'MainPID', '--value'], text=True, capture_output=True, check=True
    ).stdout.strip()
    remote = subprocess.run(
        ['ssh', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes', '02-evo-x3-tb',
         'systemctl --user show ds4-speed-001-worker.service -p MainPID --value'],
        text=True, capture_output=True, check=True
    ).stdout.strip()
    return {'coordinator_pid': int(local), 'worker_pid': int(remote)}


def request(token: str, prompt: str, expected: object) -> dict:
    conn = http.client.HTTPConnection('127.0.0.1', 18224, timeout=1800)
    body = {
        'model': 'deepseek-v4.1-flash', 'profile': 'document-low',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'seed': 1, 'max_tokens': 2048, 'stream': False,
    }
    started = time.monotonic()
    conn.request('POST', '/v1/chat/completions',
                 body=json.dumps(body, separators=(',', ':')).encode(),
                 headers={'Content-Type': 'application/json',
                          'Authorization': 'Bearer ' + token})
    response = conn.getresponse()
    raw = response.read()
    wall = time.monotonic() - started
    status = response.status
    conn.close()
    try:
        value = json.loads(raw)
    except Exception:
        value = {'_raw': raw.decode(errors='replace')}
    choice = (value.get('choices') or [{}])[0] if isinstance(value, dict) else {}
    message = choice.get('message') or {}
    content = message.get('content') or ''
    try:
        actual = json.loads(content)
    except Exception:
        actual = None
    return {
        'http': status, 'wall_s': wall, 'actual': actual, 'expected': expected,
        'finish_reason': choice.get('finish_reason'),
        'usage': value.get('usage') if isinstance(value, dict) else None,
        'pass': status == 200 and actual == expected and choice.get('finish_reason') == 'stop',
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    terminal = OUT / 'terminal.json'
    registry = OUT / 'registry.json'
    if terminal.exists():
        raise SystemExit('terminal exists, refusing replay')
    if registry.exists() and json.loads(registry.read_text()).get('state') == 'IN_FLIGHT':
        raise SystemExit('IN_FLIGHT exists, reconcile without replay')

    product = json.loads(Path('/home/funboy/reports/DS4-SPEED-001/product-e1/terminal.json').read_text())
    if product.get('status') != 'PASS':
        raise SystemExit('product gates are not PASS')

    documents = MANIFEST['documents']
    holdouts = MANIFEST['holdouts']
    code = next(item for item in documents if item['id'] == 'code2k-middle-explicit-v2')
    docs = next(item for item in documents if item['id'] == 'docs2k-middle-explicit-v2')
    hold_a = next(item for item in holdouts if item['id'] == 'document-holdout-a')
    hold_b = next(item for item in holdouts if item['id'] == 'document-holdout-b')
    cases = [
        ('code2k', code), ('docs2k', docs), ('holdout-a', hold_a),
        ('holdout-b', hold_b), ('code2k-confirm', code), ('docs2k-confirm', docs),
    ]
    plan = (cases * 4)[:24]
    token = (STATE / 'api-token').read_text().strip()
    started = time.time()
    resident = identity()
    rows: list[dict] = []
    state = {'schema': 'ds4-speed-001-soak-v1', 'state': 'IN_FLIGHT',
             'started_unix': started, 'required_seconds': 7200,
             'required_requests': 24, 'resident': resident, 'cases': rows}
    atomic(registry, state)

    for index, (name, case) in enumerate(plan, 1):
        current = {**state, 'cases': rows, 'request_state': 'IN_FLIGHT',
                   'request_index': index, 'request_case': name}
        atomic(registry, current)
        prompt = (ROOT / case['prompt_file']).read_text()
        row = request(token, prompt, case['expected'])
        row.update(index=index, case=name, finished_unix=time.time())
        rows.append(row)
        state = {**state, 'cases': rows, 'request_state': 'IDLE',
                 'request_index': None, 'request_case': None}
        atomic(registry, state)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not row['pass']:
            result = {**state, 'state': 'COMPLETE', 'status': 'FAIL',
                      'finished_unix': time.time(), 'elapsed_s': time.time() - started,
                      'final_resident': identity()}
            atomic(terminal, result)
            atomic(registry, result)
            raise SystemExit(1)
        if index < len(plan):
            time.sleep(300)

    elapsed = time.time() - started
    if elapsed < 7200:
        time.sleep(7200 - elapsed)
    final_resident = identity()
    stable = resident == final_resident
    result = {**state, 'state': 'COMPLETE',
              'status': 'PASS' if stable else 'FAIL_WORKER_LIFETIME',
              'finished_unix': time.time(), 'elapsed_s': time.time() - started,
              'final_resident': final_resident, 'worker_lifetime_pass': stable}
    atomic(terminal, result)
    atomic(registry, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not stable:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
