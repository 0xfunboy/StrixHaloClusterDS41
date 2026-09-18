#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

MODEL_DIR='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix'
DEFAULT_SYSTEM='You are a helpful assistant'
tok=DeepseekV41Tokenizer.from_pretrained(MODEL_DIR)

class H(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*a): pass
    def sendj(self,code,obj):
        b=json.dumps(obj,separators=(',',':')).encode()
        self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/health': self.sendj(200,{'status':'ok','backend':'ds4-v41-tokenizer-sidecar'});return
        self.sendj(404,{'error':'not_found'})
    def do_POST(self):
        if self.path!='/tokenize':self.sendj(404,{'error':'not_found'});return
        try:
            n=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(n))
            msgs=data.get('messages')
            if not isinstance(msgs,list):raise ValueError('messages array required')
            msgs=[dict(x) for x in msgs]
            if not msgs or msgs[0].get('role')!='system':
                msgs=[{'role':'system','content':DEFAULT_SYSTEM}]+msgs
            kw=data.get('chat_template_kwargs') or {}
            effort=kw.get('reasoning_effort') or data.get('reasoning_effort') or 'low'
            if kw.get('enable_thinking') is False: effort='none'
            ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=bool(data.get('add_generation_prompt',True)),reasoning_effort=effort)
            self.sendj(200,{'count':len(ids),'max_model_len':16384,'tokens':list(ids),'token_strs':None})
        except Exception as e:self.sendj(400,{'error':str(e)})

if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',18223),H).serve_forever()
