#!/usr/bin/env python3
from __future__ import annotations
import fcntl, http.client, json, os, signal, subprocess, time
from pathlib import Path
from typing import Any

CAMPAIGN="DS4_LONG_CONTEXT_BASELINE_001"
ROOT=Path("/home/funboy/StrixHaloClusterDS41")
RAW=Path("/home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001")
P3_MANIFEST=ROOT/"runtime/ds41/long-context-baseline-001/p3-frozen-manifest.json"
CONFIG=ROOT/"runtime/ds41/config.ds4-long-context-baseline-001.json"
NODE=ROOT/"scripts/run-ds4-long-context-baseline-001-node.sh"
GATEWAY_BIN=Path("/home/funboy/.local/share/haloclu-ds41/current/bin/strixglm")
VENV=Path("/home/funboy/StrixHaloClusterGLM/.engine/venv")
PROD_TOKEN=Path("/home/funboy/.local/state/ds4-document-profile-002/api-token")
EXP_STATE=Path("/home/funboy/.local/state/ds4-long-context-baseline-001")
EXP_TOKEN=EXP_STATE/"api-token"
LOCK=Path("/home/funboy/.local/state/strix-cluster/compute.lock")
OWNER=Path("/home/funboy/.local/state/strix-cluster/ds4-long-context-baseline-001-owner.json")
SSH=["ssh","-o","IdentityAgent=none","-o","BatchMode=yes","-o","ConnectTimeout=5","02-evo-x3-tb"]
TP_PORT=19550
API_PORT=18080
TOK_PORT=19223
GW_PORT=19224
CTX=69632
stop_requested=False

def atomic(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_suffix(path.suffix+".tmp")
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    os.replace(q,path)

def cmd(args:list[str],timeout:float|None=None,check:bool=False):
    return subprocess.run(args,text=True,capture_output=True,timeout=timeout,check=check)

def ssh(args:list[str],timeout:float|None=None,check:bool=False):
    return cmd(SSH+args,timeout=timeout,check=check)

def product_lifecycle(method="GET",action=None,body=None,timeout=30):
    token=PROD_TOKEN.read_text().strip()
    path="/v1/lifecycle"+(f"/{action}" if action else "")
    headers={"Authorization":"Bearer "+token}
    payload=None
    if body is not None:
        headers["Content-Type"]="application/json"
        payload=json.dumps(body,separators=(",",":")).encode()
    c=http.client.HTTPConnection("127.0.0.1",18224,timeout=timeout)
    c.request(method,path,body=payload,headers=headers)
    r=c.getresponse(); raw=r.read(); status=r.status; c.close()
    try: val=json.loads(raw)
    except Exception: val={"_raw":raw.decode(errors="replace")}
    return status,val

def wait_product(state:str,timeout_s:int):
    end=time.time()+timeout_s; last=None
    while time.time()<end:
        last=product_lifecycle()
        if last[0]==200 and last[1].get("state")==state: return last[1]
        time.sleep(3)
    raise RuntimeError(f"product lifecycle timeout {state}: {last}")

def product_smoke():
    token=PROD_TOKEN.read_text().strip()
    body={"model":"deepseek-v4.1-flash","profile":"document-low",
          "messages":[{"role":"user","content":"Return exactly LONGCTX-P3-RESTORE-OK."}],
          "temperature":0,"seed":1,"max_tokens":128,"stream":False}
    c=http.client.HTTPConnection("127.0.0.1",18224,timeout=1800)
    t=time.monotonic()
    c.request("POST","/v1/chat/completions",body=json.dumps(body).encode(),
              headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
    r=c.getresponse(); raw=r.read(); status=r.status; c.close()
    try: v=json.loads(raw)
    except Exception: v={"_raw":raw.decode(errors="replace")}
    ch=(v.get("choices") or [{}])[0] if isinstance(v,dict) else {}
    content=((ch.get("message") or {}).get("content") or "")
    return {"http":status,"content":content,"finish_reason":ch.get("finish_reason"),
            "wall_s":time.monotonic()-t,
            "pass":status==200 and content.strip()=="LONGCTX-P3-RESTORE-OK" and ch.get("finish_reason")=="stop"}

def port_health(port:int,path:str="/health",timeout=3):
    try:
        c=http.client.HTTPConnection("127.0.0.1",port,timeout=timeout)
        c.request("GET",path); r=c.getresponse(); raw=r.read(); s=r.status; c.close()
        return s,raw.decode(errors="replace")
    except Exception as e:
        return 0,str(e)

def wait_http(port:int,path:str,wanted=200,timeout_s=1800):
    end=time.time()+timeout_s; last=None
    while time.time()<end:
        last=port_health(port,path,3)
        if last[0]==wanted: return last
        time.sleep(3)
    raise RuntimeError(f"HTTP wait {port}{path} failed: {last}")

def site_env():
    site=cmd([str(VENV/"bin/python"),"-c","import site; print(site.getsitepackages()[0])"],check=True).stdout.strip()
    ld=":".join(["/usr/lib/x86_64-linux-gnu",f"{site}/_rocm_sdk_core/lib",f"{site}/_rocm_sdk_devel/lib",f"{site}/_rocm_sdk_libraries/lib",f"{site}/torch/lib"])
    py=":".join([str(ROOT),str(ROOT/".vendor/vllm-dsv41"),f"{site}/_rocm_sdk_core/share/amd_smi"])
    return ld,py

def reset_stop(unit:str,remote=False):
    fn=ssh if remote else cmd
    try: fn(["systemctl","--user","stop",unit],timeout=60)
    except Exception: pass
    try: fn(["systemctl","--user","reset-failed",unit],timeout=20)
    except Exception: pass

def start_runtime():
    runtime=RAW/"p3-runtime"; runtime.mkdir(parents=True,exist_ok=True)
    ssh(["mkdir","-p",str(runtime)],timeout=20,check=True)
    for u,remote in [("d4lc-p3-worker.service",True),("d4lc-p3-server.service",False),("d4lc-p3-tokenizer.service",False),("d4lc-p3-gateway.service",False)]:
        reset_stop(u,remote)

    w=["systemd-run","--user","--unit=d4lc-p3-worker","--property=KillMode=control-group",
       "--property=Restart=no","--property=TimeoutStopSec=30",
       f"--property=StandardOutput=append:{runtime}/worker.log",
       f"--property=StandardError=append:{runtime}/worker.log",
       str(NODE),"worker",str(TP_PORT),str(API_PORT),str(CTX)]
    p=ssh(w,timeout=30)
    if p.returncode: raise RuntimeError("worker start: "+(p.stderr or p.stdout))
    time.sleep(2)
    s=["systemd-run","--user","--unit=d4lc-p3-server","--property=KillMode=control-group",
       "--property=Restart=no","--property=TimeoutStopSec=30",
       f"--property=StandardOutput=append:{runtime}/server.log",
       f"--property=StandardError=append:{runtime}/server.log",
       str(NODE),"server",str(TP_PORT),str(API_PORT),str(CTX)]
    p=cmd(s,timeout=30)
    if p.returncode: raise RuntimeError("server start: "+(p.stderr or p.stdout))
    wait_http(API_PORT,"/v1/models",200,1800)

    ld,py=site_env()
    tok=["systemd-run","--user","--unit=d4lc-p3-tokenizer","--property=KillMode=control-group",
         "--property=Restart=no","--property=TimeoutStopSec=20",
         f"--property=StandardOutput=append:{runtime}/tokenizer.log",
         f"--property=StandardError=append:{runtime}/tokenizer.log",
         f"--setenv=LD_LIBRARY_PATH={ld}",f"--setenv=PYTHONPATH={py}",
         str(VENV/"bin/python"),str(ROOT/"scripts/serve-ds4-long-context-baseline-001-tokenizer.py")]
    p=cmd(tok,timeout=30)
    if p.returncode: raise RuntimeError("tokenizer start: "+(p.stderr or p.stdout))
    wait_http(TOK_PORT,"/health",200,120)

    gw=["systemd-run","--user","--unit=d4lc-p3-gateway","--property=KillMode=control-group",
        "--property=Restart=no","--property=TimeoutStopSec=20",
        f"--property=StandardOutput=append:{runtime}/gateway.log",
        f"--property=StandardError=append:{runtime}/gateway.log",
        str(GATEWAY_BIN),"serve","--config",str(CONFIG)]
    p=cmd(gw,timeout=30)
    if p.returncode: raise RuntimeError("gateway start: "+(p.stderr or p.stdout))
    end=time.time()+120
    while time.time()<end:
        if EXP_TOKEN.exists():
            try:
                token=EXP_TOKEN.read_text().strip()
                c=http.client.HTTPConnection("127.0.0.1",GW_PORT,timeout=3)
                c.request("GET","/v1/options",headers={"Authorization":"Bearer "+token})
                r=c.getresponse(); raw=r.read(); st=r.status; c.close()
                if st==200:
                    return {"options":json.loads(raw)}
            except Exception:
                pass
        time.sleep(2)
    raise RuntimeError("experimental gateway did not become ready")

def stop_runtime():
    for u,remote in [("d4lc-p3-gateway.service",False),("d4lc-p3-tokenizer.service",False),("d4lc-p3-server.service",False),("d4lc-p3-worker.service",True)]:
        reset_stop(u,remote)

def parse_json_exact(content:str):
    s=content.strip()
    try: obj=json.loads(s)
    except Exception: return None
    return obj if isinstance(obj,dict) else None

def run_stream(case_id:str,prompt:str,expected:dict):
    outdir=RAW/"p3"/case_id; outdir.mkdir(parents=True,exist_ok=True)
    term=outdir/"terminal.json"
    if term.exists():
        prior=json.loads(term.read_text())
        if prior.get("state")=="COMPLETE": return prior
    token=EXP_TOKEN.read_text().strip()
    payload={"model":"deepseek-v4.1-flash","profile":"document-long-low",
             "messages":[{"role":"user","content":prompt}],
             "reasoning_effort":"low","temperature":0,"seed":1,
             "max_tokens":2048,"context_tokens":65536,"stream":True}
    atomic(outdir/"request.json",payload)
    start=time.monotonic(); first_any=None; first_final=None
    content=""; reasoning=""; finish=None; usage=None; events=[]; done=False
    c=http.client.HTTPConnection("127.0.0.1",GW_PORT,timeout=7200)
    c.request("POST","/v1/chat/completions",body=json.dumps(payload,separators=(",",":")).encode(),
              headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","X-HaloClu-Timings":"1"})
    r=c.getresponse(); status=r.status
    headers={k.lower():v for k,v in r.getheaders()}
    rawf=(outdir/"response.sse").open("wb")
    current_event=""
    while True:
        line=r.readline()
        if not line: break
        rawf.write(line); rawf.flush()
        txt=line.decode(errors="replace").rstrip("\r\n")
        if txt.startswith("event:"):
            current_event=txt[6:].strip()
        elif txt.startswith("data:"):
            data=txt[5:].strip()
            if data=="[DONE]":
                done=True; continue
            try: obj=json.loads(data)
            except Exception: continue
            if current_event=="haloclu.timing":
                events.append({"event":current_event,"data":obj})
            choices=obj.get("choices") if isinstance(obj,dict) else None
            if isinstance(choices,list):
                for ch in choices:
                    delta=ch.get("delta") or {}
                    rc=delta.get("reasoning") or delta.get("reasoning_content") or ""
                    cc=delta.get("content") or ""
                    if (rc or cc) and first_any is None: first_any=time.monotonic()-start
                    if cc and first_final is None: first_final=time.monotonic()-start
                    reasoning+=rc; content+=cc
                    if ch.get("finish_reason"): finish=ch.get("finish_reason")
            if isinstance(obj,dict) and obj.get("usage"): usage=obj["usage"]
        elif txt=="":
            current_event=""
    rawf.close(); c.close()
    wall=time.monotonic()-start
    parsed=parse_json_exact(content)
    cached=None; prompt_tokens=None
    if isinstance(usage,dict):
        prompt_tokens=usage.get("prompt_tokens")
        det=usage.get("prompt_tokens_details") or {}
        cached=det.get("cached_tokens")
    header_prompt=headers.get("x-strixglm-prompt-tokens")
    semantic=(parsed==expected and finish=="stop")
    cache0=(cached==0)
    if status!=200 or not done:
        verdict="TECHNICAL_STOP"
    elif not content.strip() and finish in ("length",None):
        verdict="INCOMPLETE_NO_FINAL"
    elif semantic and cache0:
        verdict="PASS"
    else:
        verdict="FAIL"
    result={"case_id":case_id,"state":"COMPLETE","status":verdict,"http":status,
            "headers":headers,"finish_reason":finish,"done":done,
            "content":content,"reasoning":reasoning,
            "parsed":parsed,"expected":expected,"usage":usage,
            "prompt_tokens":prompt_tokens,"header_prompt_tokens":header_prompt,
            "cached_tokens":cached,"first_any_s":first_any,
            "first_final_s":first_final,"wall_s":wall,"telemetry_events":events}
    atomic(term,result)
    return result

def restore(lockfp,reason):
    out={"reason":reason,"started_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try: stop_runtime()
    except Exception as e: out["stop_runtime_error"]=repr(e)
    try: fcntl.flock(lockfp.fileno(),fcntl.LOCK_UN)
    except Exception: pass
    try:
        st=product_lifecycle(); out["before"]={"http":st[0],"body":st[1]}
        if st[0]==200 and st[1].get("state")!="READY":
            on=product_lifecycle("POST","on",{"confirm":True}); out["on"]={"http":on[0],"body":on[1]}
        out["ready"]=wait_product("READY",1500)
        out["smoke"]=product_smoke()
        out["status"]="PASS" if out["smoke"]["pass"] else "FAIL"
    except Exception as e:
        out["status"]="FAIL"; out["error"]=f"{type(e).__name__}: {e}"
    out["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic(RAW/"p3-restore.json",out)
    return out

def sig(signum,frame):
    global stop_requested
    stop_requested=True

def main():
    signal.signal(signal.SIGTERM,sig); signal.signal(signal.SIGINT,sig)
    terminal=RAW/"p3-terminal.json"
    if terminal.exists():
        print(terminal.read_text()); return
    frozen=json.loads(P3_MANIFEST.read_text())
    if not frozen.get("originals"): raise RuntimeError("empty P3 manifest")
    initial=product_lifecycle()
    if initial[0]!=200 or initial[1].get("state")!="READY":
        raise RuntimeError(f"product not READY: {initial}")
    atomic(RAW/"p3-registry.json",{"campaign":CAMPAIGN,"phase":"P3","state":"IN_FLIGHT",
          "started_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"cases":[]})
    off=product_lifecycle("POST","off",{"confirm":True})
    off_state=wait_product("OFF",360)
    atomic(RAW/"p3-product-off.json",{"reply":off,"state":off_state})
    lockfp=LOCK.open("a+")
    fcntl.flock(lockfp.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(OWNER,{"campaign":CAMPAIGN,"phase":"P3","state":"IN_FLIGHT","pid":os.getpid(),
                  "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"raw":str(RAW)})
    cases=[]; err=None; rt=None
    try:
        rt=start_runtime()
        atomic(RAW/"p3-runtime-ready.json",rt)
        for row in frozen["originals"]:
            if stop_requested: raise RuntimeError("stop requested")
            prompt=(ROOT/row["prompt_file"]).read_text()
            res=run_stream(row["id"],prompt,row["expected"])
            cases.append(res)
            atomic(RAW/"p3-registry.json",{"campaign":CAMPAIGN,"phase":"P3_ORIGINALS","state":"IN_FLIGHT",
                   "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"cases":cases})
            if res["status"]=="TECHNICAL_STOP":
                break
        passed={}
        for label in ("4k","8k","16k","32k","64k"):
            got=[x for x in cases if x["case_id"] in (f"code-{label}",f"docs-{label}")]
            passed[label]=(len(got)==2 and all(x["status"]=="PASS" for x in got))
        both=[x for x in ("4k","8k","16k","32k","64k") if passed.get(x)]
        max_level=both[-1] if both else None
        holdouts=[]
        if max_level and not any(x["status"]=="TECHNICAL_STOP" for x in cases):
            for row in frozen["holdouts"]:
                if row["label"]!=max_level: continue
                if stop_requested: raise RuntimeError("stop requested")
                prompt=(ROOT/row["prompt_file"]).read_text()
                res=run_stream(row["id"],prompt,row["expected"])
                holdouts.append(res)
                if res["status"]=="TECHNICAL_STOP": break
        qualified=max_level if max_level and len(holdouts)==2 and all(x["status"]=="PASS" for x in holdouts) else None
    except Exception as e:
        err=f"{type(e).__name__}: {e}"
        max_level=None; holdouts=[]; qualified=None
    finally:
        rest=restore(lockfp,err or "P3_COMPLETE")
        lockfp.close()
    status="PASS" if err is None and rest.get("status")=="PASS" else "PARTIAL_OR_STOP"
    result={"schema":"ds4-long-context-baseline-001-p3-v1","campaign":CAMPAIGN,
            "state":"COMPLETE","status":status,"error":err,"cases":cases,
            "max_both_originals":max_level,"holdouts":holdouts,
            "qualified_level":qualified,"restore":rest,
            "finished_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic(terminal,result); atomic(RAW/"p3-registry.json",result)
    atomic(OWNER,{"campaign":CAMPAIGN,"phase":"P3","state":"COMPLETE",
                  "qualified_level":qualified,"restore":rest.get("status"),
                  "updated_at":result["finished_at"]})
    print(json.dumps({"status":status,"qualified_level":qualified,
                      "cases":len(cases),"holdouts":len(holdouts),"restore":rest.get("status")},indent=2))

if __name__=="__main__":
    main()
