#!/usr/bin/env python3
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

MODEL_DIR = '/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix'
MAX_MODEL_LEN = 16384
tok = DeepseekV41Tokenizer.from_pretrained(MODEL_DIR)

BOS = '<｜begin▁of▁sentence｜>'
SYSTEM = '<｜System｜>'
USER = '<｜User｜>'
ASSISTANT = '<｜Assistant｜>'
EOS = '<｜end▁of▁sentence｜>'

def _bool_thinking(v, current: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        t = v.get('type')
        if t == 'enabled':
            return True
        if t == 'disabled':
            return False
    return current

def resolve_think_mode(data: dict) -> tuple[str, bool, int | None]:
    # Match ds4_server.c named-mode semantics for the controls this sidecar exposes.
    # Server defaults to thinking enabled/high. JSON object order determines which
    # repeated control is applied last; normal gateway payloads use consistent values.
    enabled = True
    effort = 'high'
    for key, value in data.items():
        if key in ('thinking', 'think'):
            enabled = _bool_thinking(value, enabled)
        elif key == 'reasoning_effort' and isinstance(value, str):
            effort = value
        elif key == 'chat_template_kwargs' and isinstance(value, dict):
            if 'enable_thinking' in value and isinstance(value['enable_thinking'], bool):
                enabled = value['enable_thinking']
            if isinstance(value.get('reasoning_effort'), str):
                effort = value['reasoning_effort']
    aliases = {'xhigh': 'high', 'minimal': 'low'}
    effort = aliases.get(effort, effort)
    if effort not in {'none', 'low', 'medium', 'high', 'max'}:
        raise ValueError(f'unsupported reasoning_effort {effort!r}')
    if not enabled or effort == 'none':
        return 'none', False, None
    # DeepSeek V4.1 named LOW/MEDIUM are enabled named enums but are not numeric
    # DS4_THINK_LEVEL_BASE modes. Only named high/max render numeric effort text.
    numeric = 75 if effort == 'high' else 100 if effort == 'max' else None
    return effort, True, numeric

def render_deepseek41_server(messages: list[dict], data: dict) -> tuple[str, dict]:
    mode, think, numeric = resolve_think_mode(data)
    effort_text = ''
    if numeric is not None:
        effort_text = f'Reasoning Effort: {numeric} (range 1-100, the higher the value, the more thorough the reasoning)\n\n'
    out = BOS
    initial_system = False
    if effort_text:
        out += SYSTEM + effort_text
        initial_system = True
    pending = False
    user_open = False
    last_user = -1
    for i, m in enumerate(messages):
        role = str(m.get('role') or 'user')
        if role in ('user', 'tool', 'function') or (i > 0 and role in ('system', 'developer')):
            last_user = i
    for i, m in enumerate(messages):
        role = str(m.get('role') or 'user')
        content = m.get('content')
        if not isinstance(content, str):
            raise ValueError('tokenizer sidecar currently supports text message content only')
        if role in ('system', 'developer'):
            if not (i == 0 and initial_system):
                out += SYSTEM
            out += content
            pending = i > 0
            user_open = False
        elif role == 'user':
            out += ('\n\n' if user_open else USER) + content
            pending = True
            user_open = True
        elif role == 'assistant':
            if pending:
                out += ASSISTANT
                # No tool context in this sidecar. For a historical assistant turn
                # before the last user, server closes an empty reasoning block.
                if think and i > last_user:
                    out += '<think>' + str(m.get('reasoning') or '')
                out += '</think>'
            out += content + EOS
            pending = False
            user_open = False
        else:
            raise ValueError(f'unsupported role {role!r}')
    if pending and bool(data.get('add_generation_prompt', True)):
        out += ASSISTANT + ('<think>' if think else '</think>')
    return out, {'mode': mode, 'thinking_enabled': think, 'numeric_level': numeric}

def tokenize_request(data: dict) -> dict:
    msgs = data.get('messages')
    if not isinstance(msgs, list):
        raise ValueError('messages array required')
    rendered, resolved = render_deepseek41_server([dict(x) for x in msgs], data)
    ids = list(tok.encode(rendered, add_special_tokens=False))
    return {
        'count': len(ids),
        'max_model_len': MAX_MODEL_LEN,
        'tokens': ids,
        'token_strs': None,
        'resolved_thinking': resolved,
    }

class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args): pass
    def sendj(self, code, obj):
        b = json.dumps(obj, separators=(',', ':')).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == '/health':
            self.sendj(200, {'status':'ok','backend':'ds4-v41-tokenizer-sidecar','renderer':'ds4-server-deepseek41'})
            return
        self.sendj(404, {'error':'not_found'})
    def do_POST(self):
        if self.path != '/tokenize':
            self.sendj(404, {'error':'not_found'}); return
        try:
            n = int(self.headers.get('Content-Length', '0'))
            data = json.loads(self.rfile.read(n))
            self.sendj(200, tokenize_request(data))
        except Exception as e:
            self.sendj(400, {'error': str(e)})

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 18223), H).serve_forever()
