#!/usr/bin/env python3
"""One DS41 OpenAI-chat request with raw SSE, usage and timing receipts."""
from __future__ import annotations
import argparse, json, time, urllib.request
from pathlib import Path

MODEL='DeepSeek-V4.1-Flash-MixedQ2-Engram2'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url', default='http://10.55.0.1:18210')
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--label', required=True)
    args=ap.parse_args()
    payload={
      'model':MODEL,
      'messages':[{'role':'user','content':args.prompt}],
      'temperature':args.temperature,
      'max_tokens':args.max_tokens,
      'stream':True,
      'stream_options':{'include_usage':True},
      'seed':1,
    }
    request=urllib.request.Request(
        args.url.rstrip('/')+'/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'}, method='POST')
    start=time.monotonic(); first_payload=None; first_content=None
    events=[]; text=[]; usage=None; metrics=None; finish_reason=None; request_id=None
    with urllib.request.urlopen(request, timeout=900) as response:
        status=response.status
        for raw in response:
            now=time.monotonic()
            line=raw.decode('utf-8','replace').strip()
            if not line.startswith('data: '):
                continue
            data=line[6:]
            if data == '[DONE]':
                events.append({'t_ms':(now-start)*1000,'done':True})
                break
            obj=json.loads(data); events.append({'t_ms':(now-start)*1000,'data':obj})
            request_id=request_id or obj.get('id')
            if first_payload is None: first_payload=now
            if obj.get('usage') is not None: usage=obj['usage']
            if obj.get('metrics') is not None: metrics=obj['metrics']
            for choice in obj.get('choices') or []:
                if choice.get('finish_reason') is not None: finish_reason=choice['finish_reason']
                delta=choice.get('delta') or {}
                piece=delta.get('content')
                if piece:
                    if first_content is None: first_content=now
                    text.append(piece)
    end=time.monotonic(); content=''.join(text)
    prompt_tokens=(usage or {}).get('prompt_tokens')
    completion_tokens=(usage or {}).get('completion_tokens')
    engine_ttft=(metrics or {}).get('time_to_first_token_ms')
    generation_ms=(metrics or {}).get('generation_time_ms')
    decode_tps=None
    if generation_ms and completion_tokens and completion_tokens>1:
        decode_tps=(completion_tokens-1)/(generation_ms/1000)
    prompt_ttft_tps=None
    if engine_ttft and prompt_tokens:
        prompt_ttft_tps=prompt_tokens/(engine_ttft/1000)
    result={
      'label':args.label,'status_code':status,'request_id':request_id,
      'request':payload,'content':content,'finish_reason':finish_reason,
      'usage':usage,'metrics':metrics,
      'client':{
        'tt_first_sse_ms': None if first_payload is None else (first_payload-start)*1000,
        'tt_first_content_ms': None if first_content is None else (first_content-start)*1000,
        'http_e2e_ms':(end-start)*1000,
      },
      'derived':{
        'decode_tps_first_to_last':decode_tps,
        'prompt_over_engine_ttft_tps':prompt_ttft_tps,
        'prompt_over_engine_ttft_note':'Effective prompt/TTFT throughput; TTFT includes first-token work, not pure kernel prefill.',
      },
      'events':events,
    }
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('label','status_code','finish_reason','usage','metrics','client','derived')},indent=2,ensure_ascii=False))
    print('CONTENT_BEGIN'); print(content); print('CONTENT_END')

if __name__=='__main__': main()
