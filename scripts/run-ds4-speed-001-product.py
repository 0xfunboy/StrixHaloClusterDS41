#!/usr/bin/env python3
from __future__ import annotations

import http.client
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
OUT = Path('/home/funboy/reports/DS4-SPEED-001/product-e1')
STATE = Path('/home/funboy/.local/state/ds4-document-profile-002')
BASE_CONFIG = ROOT / 'runtime/ds41/config.ds4-document-profile-002.json'
E1_CONFIG = ROOT / 'runtime/ds41/config.ds4-speed-001-engram1.json'
TOKEN_FILE = STATE / 'api-token'
GATEWAY = ('127.0.0.1', 18224)


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def request(method: str, path: str, token: str | None = None,
            body: dict | None = None, timeout: int = 1800) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(*GATEWAY, timeout=timeout)
    headers = {}
    payload = None
    if body is not None:
        payload = json.dumps(body, separators=(',', ':')).encode()
        headers['Content-Type'] = 'application/json'
    if token:
        headers['Authorization'] = 'Bearer ' + token
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    status = response.status
    conn.close()
    try:
        value = json.loads(raw)
    except Exception:
        value = {'_raw': raw.decode(errors='replace')}
    return status, value


def chat(token: str, messages: list[dict], expected: str) -> dict:
    body = {
        'model': 'deepseek-v4.1-flash',
        'profile': 'document-low',
        'messages': messages,
        'temperature': 0,
        'seed': 1,
        'max_tokens': 128,
        'stream': False,
    }
    started = time.monotonic()
    status, value = request('POST', '/v1/chat/completions', token, body)
    wall = time.monotonic() - started
    choice = (value.get('choices') or [{}])[0] if isinstance(value, dict) else {}
    message = choice.get('message') or {}
    content = message.get('content') or ''
    return {
        'http': status,
        'content': content,
        'expected': expected,
        'finish_reason': choice.get('finish_reason'),
        'wall_s': wall,
        'usage': value.get('usage') if isinstance(value, dict) else None,
        'pass': status == 200 and content.strip() == expected and
                choice.get('finish_reason') == 'stop',
    }


def stream(token: str, body: dict, cancel_after_first: bool = False) -> dict:
    conn = http.client.HTTPConnection(*GATEWAY, timeout=1800)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token,
    }
    conn.request('POST', '/v1/chat/completions',
                 body=json.dumps(body, separators=(',', ':')).encode(),
                 headers=headers)
    response = conn.getresponse()
    result = {'http': response.status, 'content': '', 'reasoning': '',
              'finish_reason': None, 'events': 0, 'cancelled': False}
    if response.status != 200:
        result['error'] = response.read().decode(errors='replace')
        conn.close()
        return result
    frame: list[str] = []
    while True:
        raw = response.readline()
        if not raw:
            break
        line = raw.decode(errors='replace').rstrip('\r\n')
        if line:
            frame.append(line)
            continue
        data = [item[5:].lstrip() for item in frame if item.startswith('data:')]
        frame = []
        if not data:
            continue
        text = '\n'.join(data)
        if text == '[DONE]':
            break
        try:
            value = json.loads(text)
        except Exception:
            continue
        result['events'] += 1
        for choice in value.get('choices') or []:
            delta = choice.get('delta') or {}
            result['content'] += delta.get('content') or ''
            result['reasoning'] += delta.get('reasoning_content') or delta.get('reasoning') or ''
            if choice.get('finish_reason') is not None:
                result['finish_reason'] = choice['finish_reason']
        if cancel_after_first:
            result['cancelled'] = True
            conn.close()
            return result
    conn.close()
    return result


def lifecycle(token: str, action: str | None = None) -> tuple[int, dict]:
    if action:
        return request('POST', '/v1/lifecycle/' + action, token, {'confirm': True}, 30)
    return request('GET', '/v1/lifecycle', token, timeout=30)


def wait_state(token: str, wanted: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    last: tuple[int, dict] | None = None
    while time.time() < deadline:
        last = lifecycle(token)
        if last[0] == 200 and last[1].get('state') == wanted:
            return last[1]
        time.sleep(3)
    raise RuntimeError(f'lifecycle timeout for {wanted}: {last}')


def service_identity() -> dict:
    local = subprocess.run(
        ['systemctl', '--user', 'show', 'ds4-speed-001-coordinator.service',
         '-p', 'MainPID', '--value'], text=True, capture_output=True, check=True
    ).stdout.strip()
    remote = subprocess.run(
        ['ssh', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes', '02-evo-x3-tb',
         'systemctl --user show ds4-speed-001-worker.service -p MainPID --value'],
        text=True, capture_output=True, check=True
    ).stdout.strip()
    local_exe = os.path.realpath(f'/proc/{local}/exe') if local != '0' else ''
    remote_exe = subprocess.run(
        ['ssh', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes', '02-evo-x3-tb',
         f'readlink -f /proc/{remote}/exe'], text=True, capture_output=True, check=True
    ).stdout.strip() if remote != '0' else ''
    local_env = Path(f'/proc/{local}/environ').read_bytes().split(b'\0') if local != '0' else []
    remote_env = subprocess.run(
        ['ssh', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes', '02-evo-x3-tb',
         f"tr '\\0' '\\n' </proc/{remote}/environ | grep '^DS4_' || true"],
        text=True, capture_output=True, check=True
    ).stdout.splitlines() if remote != '0' else []
    return {
        'coordinator_pid': int(local),
        'worker_pid': int(remote),
        'coordinator_exe': local_exe,
        'worker_exe': remote_exe,
        'coordinator_ds4_env': sorted(
            item.decode(errors='replace') for item in local_env if item.startswith(b'DS4_')),
        'worker_ds4_env': sorted(remote_env),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    terminal = OUT / 'terminal.json'
    registry = OUT / 'registry.json'
    if terminal.exists():
        raise SystemExit('terminal exists, refusing replay')
    if registry.exists():
        prior = json.loads(registry.read_text())
        if prior.get('state') == 'IN_FLIGHT':
            raise SystemExit('IN_FLIGHT exists, reconcile without replay')

    base = json.loads(BASE_CONFIG.read_text())
    e1 = json.loads(E1_CONFIG.read_text())
    base_lifecycle = base.pop('lifecycle_command')
    e1_lifecycle = e1.pop('lifecycle_command')
    config_pass = base == e1 and base_lifecycle.endswith('ds4-document-controller.sh') and \
        e1_lifecycle.endswith('ds4-speed-001-controller.sh')

    token = TOKEN_FILE.read_text().strip()
    rows: list[dict] = []
    atomic(registry, {'schema': 'ds4-speed-001-product-v1', 'state': 'IN_FLIGHT',
                      'started_unix': time.time(), 'gates': rows})

    def add(name: str, passed: bool, detail: object) -> None:
        rows.append({'gate': name, 'pass': bool(passed), 'detail': detail})
        atomic(registry, {'schema': 'ds4-speed-001-product-v1', 'state': 'IN_FLIGHT',
                          'started_unix': started, 'gates': rows})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    started = time.time()
    add('profile_config_unchanged', config_pass,
        {'base_lifecycle': base_lifecycle, 'e1_lifecycle': e1_lifecycle})
    before = service_identity()
    release = '/releases/ds4-speed-001-engram1/'
    forbidden = ('DS4_V41_ENGRAM_TIMING=', 'DS4_V41_DISABLE_ENGRAM_CONCURRENT=',
                 'DS4_ROCM_V41_VERIFY2=')
    env_lines = before['coordinator_ds4_env'] + before['worker_ds4_env']
    identity_pass = release in before['coordinator_exe'] and release in before['worker_exe']
    env_pass = not any(line.startswith(forbidden) for line in env_lines)
    add('resident_e1_identity', identity_pass and env_pass, before)

    status, _ = request('POST', '/v1/chat/completions', None,
                        {'messages': [{'role': 'user', 'content': 'x'}]}, 30)
    add('unauthorized_chat', status == 401, {'http': status})
    status, state = lifecycle(token)
    add('lifecycle_ready', status == 200 and state.get('state') == 'READY',
        {'http': status, 'body': state})

    add('gateway_nonstream', **({'passed': (result := chat(token,
        [{'role': 'user', 'content': 'Compute 23+19. Return only the integer.'}], '42'))['pass'],
        'detail': result}))

    stream_body = {
        'model': 'deepseek-v4.1-flash', 'profile': 'document-low',
        'messages': [{'role': 'user', 'content': 'Compute 31+11. Return only the integer.'}],
        'temperature': 0, 'seed': 1, 'max_tokens': 128, 'stream': True,
    }
    result = stream(token, stream_body)
    add('sse_reasoning_final', result['http'] == 200 and
        result['content'].strip() == '42' and result['finish_reason'] == 'stop', result)

    result = chat(token, [{'role': 'user', 'content': 'Return exactly FRESH-OK.'}], 'FRESH-OK')
    add('fresh_request', result['pass'], result)
    first = chat(token, [{'role': 'user', 'content':
        'Remember the code ALPHA-731. Reply only READY.'}], 'READY')
    second = chat(token, [
        {'role': 'user', 'content': 'Remember the code ALPHA-731. Reply only READY.'},
        {'role': 'assistant', 'content': 'READY'},
        {'role': 'user', 'content': 'Return only the code I asked you to remember.'},
    ], 'ALPHA-731')
    add('multiturn_history', first['pass'] and second['pass'],
        {'turn1': first, 'turn2': second})

    cancel_body = {
        'model': 'deepseek-v4.1-flash', 'profile': 'document-low',
        'messages': [{'role': 'user', 'content':
                      'List the integers from 1 through 500, one per line.'}],
        'temperature': 0, 'seed': 1, 'max_tokens': 2048, 'stream': True,
    }
    cancelled = stream(token, cancel_body, cancel_after_first=True)
    deadline = time.time() + 300
    resumed = None
    while time.time() < deadline:
        resumed = chat(token, [{'role': 'user', 'content':
                                'Return exactly DRAIN-OK.'}], 'DRAIN-OK')
        if resumed['pass']:
            break
        time.sleep(3)
    add('cancel_drain_resume', cancelled.get('cancelled') and
        bool(resumed and resumed['pass']), {'cancel': cancelled, 'resume': resumed})

    stable = service_identity()
    add('worker_lifetime_across_requests',
        before['coordinator_pid'] == stable['coordinator_pid'] and
        before['worker_pid'] == stable['worker_pid'], {'before': before, 'after': stable})

    off_status, off_reply = lifecycle(token, 'off')
    off_state = wait_state(token, 'OFF', 180)
    unavailable = chat(token, [{'role': 'user', 'content':
                                'Return exactly SHOULD-NOT-RUN.'}], 'SHOULD-NOT-RUN')
    add('whole_pair_off_no_autoload', off_status == 202 and
        unavailable['http'] == 503, {'reply': off_reply, 'state': off_state,
                                     'chat': unavailable})

    on_status, on_reply = lifecycle(token, 'on')
    on_state = wait_state(token, 'READY', 1300)
    after = service_identity()
    post = chat(token, [{'role': 'user', 'content': 'Return exactly ON-OK.'}], 'ON-OK')
    restarted = after['coordinator_pid'] != before['coordinator_pid'] and \
        after['worker_pid'] != before['worker_pid']
    identity_after = release in after['coordinator_exe'] and release in after['worker_exe']
    env_after = not any(line.startswith(forbidden) for line in
                        after['coordinator_ds4_env'] + after['worker_ds4_env'])
    add('whole_pair_on_and_resume', on_status == 202 and post['pass'] and
        restarted and identity_after and env_after,
        {'reply': on_reply, 'state': on_state, 'identity': after, 'chat': post})

    passed = all(row['pass'] for row in rows)
    result = {'schema': 'ds4-speed-001-product-v1',
              'status': 'PASS' if passed else 'FAIL',
              'started_unix': started, 'finished_unix': time.time(), 'gates': rows}
    atomic(terminal, result)
    atomic(registry, {**result, 'state': 'COMPLETE'})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
