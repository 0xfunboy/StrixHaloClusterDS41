#!/usr/bin/env python3
import json, time, argparse
import torch
ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
assert torch.cuda.is_available(); torch.cuda.set_device(0)
stream=torch.cuda.current_stream(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
start.record(stream)
# Simulate prefill target work and host-side gap between chunks without per-op sync.
x=torch.randn((1024,1024),device='cuda',dtype=torch.float16); y=x@x
# Host gap after start event should remain within event-to-event wall/device timeline.
time.sleep(0.02)
z=y.relu(); end.record(stream)
# Simulate sampling after the end marker; one later blocking scalar makes prior event complete.
w=z.sum(); _=float(w.cpu())
assert end.query(), 'end event unexpectedly incomplete after later blocking output copy'
ms=float(start.elapsed_time(end)); assert ms>0
out={'status':'PASS','rank':a.rank,'device':torch.cuda.get_device_name(0),'gcn':getattr(torch.cuda.get_device_properties(0),'gcnArchName',None),'prefill_event_ms':ms,'end_query_after_later_sync':True,'sampling_order':'end event recorded before post-marker work'}
open(a.output,'w').write(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
