#!/usr/bin/env python3
from __future__ import annotations
import fcntl, hashlib, http.client, json, os, signal, subprocess, time
from pathlib import Path
from typing import Any

CAMPAIGN="DS4_PREFILL_GAP_002"
ROOT=Path("/home/funboy/StrixHaloClusterDS41")
RAW=Path("/home/funboy/reports/DS4-PREFILL-GAP-002")
ART=RAW/"artifacts/B"
MODEL=Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
PRE=RAW/"quality-preregister.json"
CONFIG=ROOT/"runtime/ds41/config.ds4-prefill-gap-002-quality.json"
TOKENIZER_RUNNER=ROOT/"scripts/serve-ds4-prefill-gap-002-tokenizer.py"
STRIX=Path("/home/funboy/.local/share/haloclu-ds41/current/bin/strixglm")
VENV=Path("/home/funboy/StrixHaloClusterGLM/.engine/venv")
PROD_TOKEN=Path("/home/funboy/.local/state/ds4-document-profile-002/api-token")
EXP_STATE=Path("/home/funboy/.local/state/ds4-prefill-gap-002-quality")
EXP_TOKEN=EXP_STATE/"api-token"
LOCK=Path("/home/funboy/.local/state/strix-cluster/compute.lock")
OWNER=Path("/home/funboy/.local/state/strix-cluster/ds4-prefill-gap-002-owner.json")
REGISTRY=RAW/"quality-registry.json"
TERMINAL=RAW/"quality-terminal.json"
RESTORE=RAW/"restore-after-quality.json"
PROD=("127.0.0.1",18224); GW=("127.0.0.1",19424); TOK=("127.0.0.1",19423); BACKEND=("127.0.0.1",18084)
COORD="10.55.0.1"; CTX=69632
SSH=["ssh","-o","IdentityAgent=none","-o","BatchMode=yes","-o","ConnectTimeout=5","02-evo-x3-tb"]
tok_unit="d4pg2-quality-tokenizer"; gw_unit="d4pg2-quality-gateway"
stop_requested=False
current_pair:dict[str,Any]|None=None

def now():return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);q=path.with_suffix(path.suffix+".tmp")
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n");os.replace(q,path)
def cmd(a,timeout=None,check=False):return subprocess.run(a,text=True,capture_output=True,timeout=timeout,check=check)
def ssh(a,timeout=None,check=False):return cmd(SSH+a,timeout,check)
def shab(path):
    h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()
def site_env():
    site=cmd([str(VENV/"bin/python"),"-c","import site; print(site.getsitepackages()[0])"],check=True).stdout.strip()
    ld=":".join(["/usr/lib/x86_64-linux-gnu",f"{site}/_rocm_sdk_core/lib",f"{site}/_rocm_sdk_devel/lib",f"{site}/_rocm_sdk_libraries/lib",f"{site}/torch/lib"])
    py=":".join([str(ROOT),str(ROOT/".vendor/vllm-dsv41"),f"{site}/_rocm_sdk_core/share/amd_smi"])
    return ld,py
LD,PYTHONPATH=site_env()
UNSET=["DS4_METAL_GRAPH_PREFILL_PROFILE","DS4_TP_PREFILL_PROFILE","DS4_V41_ENGRAM_TIMING","DS4_ROCM_V41_VERIFY2","DS4_V41_DISABLE_ENGRAM_CONCURRENT"]
def clean(binary):
    a=["/usr/bin/env"]
    for k in UNSET:a+=["-u",k]
    a.append(binary);return a

def lifecycle(method="GET",action=None,body=None,timeout=30):
    t=PROD_TOKEN.read_text().strip();path="/v1/lifecycle"+(f"/{action}" if action else "")
    headers={"Authorization":"Bearer "+t};payload=None
    if body is not None:headers["Content-Type"]="application/json";payload=json.dumps(body,separators=(",",":")).encode()
    c=http.client.HTTPConnection(*PROD,timeout=timeout);c.request(method,path,body=payload,headers=headers)
    r=c.getresponse();raw=r.read();st=r.status;c.close()
    try:v=json.loads(raw)
    except:v={"_raw":raw.decode(errors="replace")}
    return st,v
def wait_prod(state,timeout_s):
    end=time.time()+timeout_s;last=None
    while time.time()<end:
        last=lifecycle()
        if last[0]==200 and last[1].get("state")==state:return last[1]
        time.sleep(3)
    raise RuntimeError(f"prod timeout {state}: {last}")
def prod_smoke(expected):
    t=PROD_TOKEN.read_text().strip()
    body={"model":"deepseek-v4.1-flash","profile":"document-low","messages":[{"role":"user","content":f"Return exactly {expected}."}],
          "temperature":0,"seed":1,"max_tokens":128,"stream":False}
    c=http.client.HTTPConnection(*PROD,timeout=1800);start=time.monotonic()
    c.request("POST","/v1/chat/completions",body=json.dumps(body).encode(),headers={"Authorization":"Bearer "+t,"Content-Type":"application/json"})
    r=c.getresponse();raw=r.read();st=r.status;c.close()
    try:v=json.loads(raw)
    except:v={}
    ch=(v.get("choices") or [{}])[0];content=((ch.get("message") or {}).get("content") or "")
    return {"http":st,"content":content,"finish_reason":ch.get("finish_reason"),"wall_s":time.monotonic()-start,
            "pass":st==200 and content.strip()==expected and ch.get("finish_reason")=="stop"}

def stop_unit(unit,remote=False):
    try:(ssh if remote else cmd)(["systemctl","--user","stop",unit+".service"],timeout=60)
    except:pass
def start_infra():
    EXP_STATE.mkdir(parents=True,exist_ok=True)
    for u in (tok_unit,gw_unit):stop_unit(u);cmd(["systemctl","--user","reset-failed",u+".service"],timeout=20)
    tp=["systemd-run","--user",f"--unit={tok_unit}","--property=KillMode=control-group","--property=Restart=no",
        "--property=TimeoutStopSec=20",f"--property=StandardOutput=append:{RAW}/quality-tokenizer.log",
        f"--property=StandardError=append:{RAW}/quality-tokenizer.log",f"--setenv=LD_LIBRARY_PATH={LD}",f"--setenv=PYTHONPATH={PYTHONPATH}",
        str(VENV/"bin/python"),str(TOKENIZER_RUNNER)]
    p=cmd(tp,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    end=time.time()+120
    while time.time()<end:
        try:
            c=http.client.HTTPConnection(*TOK,timeout=3);c.request("GET","/health");r=c.getresponse();raw=r.read();c.close()
            if r.status==200:break
        except:pass
        time.sleep(2)
    else:raise RuntimeError("tokenizer not ready")
    gp=["systemd-run","--user",f"--unit={gw_unit}","--property=KillMode=control-group","--property=Restart=no",
        "--property=TimeoutStopSec=20",f"--property=StandardOutput=append:{RAW}/quality-gateway.log",
        f"--property=StandardError=append:{RAW}/quality-gateway.log",str(STRIX),"serve","--config",str(CONFIG)]
    p=cmd(gp,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    end=time.time()+120
    while time.time()<end:
        if EXP_TOKEN.exists():
            try:
                c=http.client.HTTPConnection(*GW,timeout=3);c.request("GET","/v1/options",headers={"Authorization":"Bearer "+EXP_TOKEN.read_text().strip()})
                r=c.getresponse();raw=r.read();c.close()
                if r.status==200:return json.loads(raw)
            except:pass
        time.sleep(2)
    raise RuntimeError("gateway not ready")

def status(unit,remote=False):
    p=(ssh if remote else cmd)(["systemctl","--user","show",unit+".service","-p","ActiveState","-p","Result","-p","ExecMainStatus","-p","MainPID","--no-pager"],timeout=15)
    d={}; 
    for line in p.stdout.splitlines():
        if "=" in line:k,v=line.split("=",1);d[k]=v
    return d
def start_pair(tag,port):
    global current_pair
    lu=f"d4pg2-q-{tag}-c";ru=f"d4pg2-q-{tag}-w";d=RAW/"quality"/tag;d.mkdir(parents=True,exist_ok=True);ssh(["mkdir","-p",str(d)],timeout=20)
    for remote,u in ((False,lu),(True,ru)):
        stop_unit(u,remote);(ssh if remote else cmd)(["systemctl","--user","reset-failed",u+".service"],timeout=20)
    props=["--property=KillMode=control-group","--property=Restart=no","--property=TimeoutStopSec=45",
           f"--setenv=LD_LIBRARY_PATH={LD}","--setenv=OMP_NUM_THREADS=1","--setenv=DS4_TP_GATE_TIMEOUT_MS=5000"]
    w=["systemd-run","--user",f"--unit={ru}",*props,f"--property=StandardOutput=append:{d}/worker.log",f"--property=StandardError=append:{d}/worker.log",
       *clean(str(ART/"ds4")),"--rocm","-m",str(MODEL),"--ctx",str(CTX),"--role","worker","--coordinator",COORD,str(port),"--tensor-parallel","--transport","tcp"]
    p=ssh(w,timeout=30)
    if p.returncode:raise RuntimeError("worker start "+(p.stderr or p.stdout))
    time.sleep(2)
    s=["systemd-run","--user",f"--unit={lu}",*props,f"--property=StandardOutput=append:{d}/server.log",f"--property=StandardError=append:{d}/server.log",
       *clean(str(ART/"ds4-server")),"--rocm","-m",str(MODEL),"--ctx",str(CTX),"--role","coordinator","--listen",COORD,str(port),"--tensor-parallel","--transport","tcp",
       "--batched-session","1","--host","127.0.0.1","--port","18084"]
    p=cmd(s,timeout=30)
    if p.returncode:stop_unit(ru,True);raise RuntimeError("server start "+(p.stderr or p.stdout))
    current_pair={"local":lu,"remote":ru,"dir":str(d)}
    end=time.time()+1500
    while time.time()<end:
        if stop_requested:raise RuntimeError("stop requested")
        if status(lu).get("ActiveState")=="failed" or status(ru,True).get("ActiveState")=="failed":raise RuntimeError("pair failed")
        try:
            c=http.client.HTTPConnection(*BACKEND,timeout=3);c.request("GET","/v1/models");r=c.getresponse();r.read();c.close()
            if r.status==200:return current_pair
        except:pass
        time.sleep(5)
    raise RuntimeError("pair not ready")
def stop_pair():
    global current_pair
    if not current_pair:return
    lu=current_pair["local"];ru=current_pair["remote"]
    lp=subprocess.Popen(["systemctl","--user","stop",lu+".service"]);rp=subprocess.Popen(SSH+["systemctl","--user","stop",ru+".service"])
    try:lp.wait(60)
    except:lp.kill()
    try:rp.wait(60)
    except:rp.kill()
    current_pair=None

def run_request(item):
    d=RAW/"quality"/item["id"];d.mkdir(parents=True,exist_ok=True)
    term=d/"terminal.json"
    if term.exists():return json.loads(term.read_text())
    src=ROOT/item["source"]
    if shab(src)!=item["sha256"]:raise RuntimeError(f"source hash mismatch {item['id']}")
    prompt=src.read_text()
    body={"model":"deepseek-v4.1-flash","profile":"document-low","messages":[{"role":"user","content":prompt}],
          "reasoning_effort":"low","temperature":0,"seed":1,"max_tokens":2048,"context_tokens":65536,"stream":False}
    atomic(d/"request.json",body)
    token=EXP_TOKEN.read_text().strip();start=time.monotonic()
    c=http.client.HTTPConnection(*GW,timeout=7200)
    c.request("POST","/v1/chat/completions",body=json.dumps(body,separators=(",",":")).encode(),
              headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
    r=c.getresponse();headers={k.lower():v for k,v in r.getheaders()};raw=r.read();st=r.status;c.close();wall=time.monotonic()-start
    (d/"response.raw").write_bytes(raw)
    try:v=json.loads(raw)
    except Exception:v={"_raw":raw.decode(errors="replace")}
    ch=(v.get("choices") or [{}])[0] if isinstance(v,dict) else {};msg=ch.get("message") or {}
    content=msg.get("content") or "";reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "";finish=ch.get("finish_reason")
    try:parsed=json.loads(content.strip())
    except Exception:parsed=None
    usage=v.get("usage") if isinstance(v,dict) else None;det=(usage or {}).get("prompt_tokens_details") or {};cached=det.get("cached_tokens")
    if st!=200:result="TECHNICAL_STOP"
    elif parsed==item["expected"] and finish=="stop" and cached==0:result="PASS"
    elif finish=="length":result="INCOMPLETE_NO_FINAL"
    else:result="SEMANTIC_FAIL"
    out={"id":item["id"],"state":"COMPLETE","status":result,"http":st,"finish_reason":finish,"content":content,"reasoning":reasoning,
         "parsed":parsed,"expected":item["expected"],"usage":usage,"cached_tokens":cached,"prompt_header":headers.get("x-strixglm-prompt-tokens"),"wall_s":wall}
    atomic(term,out);return out

def restore(lockfp,reason):
    try:stop_pair()
    except:pass
    stop_unit(gw_unit);stop_unit(tok_unit)
    try:fcntl.flock(lockfp.fileno(),fcntl.LOCK_UN)
    except:pass
    out={"reason":reason,"started_at":now()}
    try:
        before=lifecycle();out["before"]={"http":before[0],"body":before[1]}
        if before[0]==200 and before[1].get("state")!="READY":out["on"]={"reply":lifecycle("POST","on",{"confirm":True})}
        out["ready"]=wait_prod("READY",1500);out["smoke"]=prod_smoke("PREFILL-GAP-QUALITY-RESTORE-OK")
        out["status"]="PASS" if out["smoke"]["pass"] else "FAIL"
    except Exception as e:out["status"]="FAIL";out["error"]=f"{type(e).__name__}: {e}"
    out["finished_at"]=now();atomic(RESTORE,out);return out
def sig(a,b):
    global stop_requested;stop_requested=True

def main():
    signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
    if TERMINAL.exists():print(TERMINAL.read_text());return
    pre=json.loads(PRE.read_text())
    initial=lifecycle()
    if initial[0]!=200 or initial[1].get("state")!="READY":raise RuntimeError("E1 not READY")
    atomic(REGISTRY,{"campaign":CAMPAIGN,"state":"IN_FLIGHT","phase":"ACQUIRE","started_at":now(),"cases":[]})
    off=lifecycle("POST","off",{"confirm":True});offstate=wait_prod("OFF",420);atomic(RAW/"gateway-off-before-quality.json",{"reply":off,"state":offstate})
    lockfp=LOCK.open("a+");fcntl.flock(lockfp.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=[];error=None
    try:
        options=start_infra();atomic(RAW/"quality-gateway-ready.json",options)
        for i,item in enumerate(pre["requests"]):
            pair=start_pair(item["id"],19900+i)
            try:r=run_request(item)
            finally:stop_pair()
            cases.append(r);atomic(REGISTRY,{"campaign":CAMPAIGN,"state":"IN_FLIGHT","phase":"QUALITY","updated_at":now(),"cases":cases})
            if r["status"]=="TECHNICAL_STOP":raise RuntimeError(f"technical stop {item['id']}")
    except Exception as e:error=f"{type(e).__name__}: {e}"
    finally:
        rest=restore(lockfp,error or "QUALITY_COMPLETE");lockfp.close()
    terminal={"campaign":CAMPAIGN,"state":"COMPLETE","status":"PASS" if error is None and rest.get("status")=="PASS" else "PARTIAL_OR_STOP",
              "error":error,"cases":cases,"restore":rest,"finished_at":now()}
    atomic(TERMINAL,terminal);atomic(REGISTRY,terminal);atomic(OWNER,{"campaign":CAMPAIGN,"state":"COMPLETE","updated_at":terminal["finished_at"],"terminal":str(TERMINAL)})
    print(json.dumps({"status":terminal["status"],"cases":[(x["id"],x["status"]) for x in cases],"restore":rest.get("status")},indent=2))
if __name__=="__main__":main()
