#!/usr/bin/env python3
from __future__ import annotations
import csv, fcntl, hashlib, http.client, json, os, re, signal, statistics, subprocess, time
from pathlib import Path
from typing import Any

CAMPAIGN="DS4_PREFILL_GAP_002"
ROOT=Path("/home/funboy/StrixHaloClusterDS41")
RAW=Path("/home/funboy/reports/DS4-PREFILL-GAP-002")
ART=RAW/"artifacts"
MODEL=Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
CORPUS=Path("/home/funboy/worktrees/ds4-prefill-gap-002/speed-bench/promessi_sposi.txt")
TOKEN=Path("/home/funboy/.local/state/ds4-document-profile-002/api-token")
LOCK=Path("/home/funboy/.local/state/strix-cluster/compute.lock")
OWNER=Path("/home/funboy/.local/state/strix-cluster/ds4-prefill-gap-002-owner.json")
REGISTRY=RAW/"ab-registry.json"
TERMINAL=RAW/"ab-terminal.json"
RESTORE=RAW/"restore-after-ab.json"
PREREG=RAW/"ab-preregister.json"
P1=RAW/"p1-terminal.json"
VENV=Path("/home/funboy/StrixHaloClusterGLM/.engine/venv")
GATEWAY=("127.0.0.1",18224)
COORD="10.55.0.1"
CTX=69632
SSH=["ssh","-o","IdentityAgent=none","-o","BatchMode=yes","-o","ConnectTimeout=5","02-evo-x3-tb"]
ORDER16=["A1","B1","B2","A2","A3","B3"]
ORDER64=["B1","A1","A2","B2","B3","A3"]
stop_requested=False
active_local:set[str]=set()
active_remote:set[str]=set()

def now(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def atomic(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_suffix(path.suffix+".tmp")
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    os.replace(q,path)
def cmd(args:list[str],timeout:float|None=None,check:bool=False):
    return subprocess.run(args,text=True,capture_output=True,timeout=timeout,check=check)
def ssh(args:list[str],timeout:float|None=None,check:bool=False):
    return cmd(SSH+args,timeout=timeout,check=check)
def sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def lifecycle(method="GET",action=None,body=None,timeout=30):
    token=TOKEN.read_text().strip()
    path="/v1/lifecycle"+(f"/{action}" if action else "")
    headers={"Authorization":"Bearer "+token}; payload=None
    if body is not None:
        headers["Content-Type"]="application/json"; payload=json.dumps(body,separators=(",",":")).encode()
    c=http.client.HTTPConnection(*GATEWAY,timeout=timeout)
    c.request(method,path,body=payload,headers=headers); r=c.getresponse(); raw=r.read(); st=r.status; c.close()
    try:v=json.loads(raw)
    except Exception:v={"_raw":raw.decode(errors="replace")}
    return st,v
def wait_lifecycle(state,timeout_s):
    end=time.time()+timeout_s; last=None
    while time.time()<end:
        last=lifecycle()
        if last[0]==200 and last[1].get("state")==state:return last[1]
        time.sleep(3)
    raise RuntimeError(f"lifecycle timeout {state}: {last}")
def smoke(expected):
    token=TOKEN.read_text().strip()
    body={"model":"deepseek-v4.1-flash","profile":"document-low",
          "messages":[{"role":"user","content":f"Return exactly {expected}."}],
          "temperature":0,"seed":1,"max_tokens":128,"stream":False}
    t=time.monotonic(); c=http.client.HTTPConnection(*GATEWAY,timeout=1800)
    c.request("POST","/v1/chat/completions",body=json.dumps(body,separators=(",",":")).encode(),
              headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
    r=c.getresponse(); raw=r.read(); st=r.status; c.close()
    try:v=json.loads(raw)
    except Exception:v={"_raw":raw.decode(errors="replace")}
    ch=(v.get("choices") or [{}])[0] if isinstance(v,dict) else {}
    content=((ch.get("message") or {}).get("content") or "")
    return {"http":st,"content":content,"finish_reason":ch.get("finish_reason"),
            "wall_s":time.monotonic()-t,
            "pass":st==200 and content.strip()==expected and ch.get("finish_reason")=="stop"}

def site_ld():
    site=cmd([str(VENV/"bin/python"),"-c","import site; print(site.getsitepackages()[0])"],check=True).stdout.strip()
    return ":".join(["/usr/lib/x86_64-linux-gnu",f"{site}/_rocm_sdk_core/lib",f"{site}/_rocm_sdk_devel/lib",
                     f"{site}/_rocm_sdk_libraries/lib",f"{site}/torch/lib"])
LD=site_ld()
UNSET=["DS4_METAL_GRAPH_PREFILL_PROFILE","DS4_TP_PREFILL_PROFILE","DS4_V41_ENGRAM_TIMING",
       "DS4_ROCM_V41_VERIFY2","DS4_V41_DISABLE_ENGRAM_CONCURRENT"]
def clean_env(binary:str)->list[str]:
    out=["/usr/bin/env"]
    for k in UNSET: out+=["-u",k]
    out.append(binary)
    return out

def status_unit(unit,remote=False):
    args=["systemctl","--user","show",unit+".service","-p","ActiveState","-p","SubState","-p","Result",
          "-p","ExecMainStatus","-p","MainPID","-p","InvocationID","--no-pager"]
    p=ssh(args,timeout=15) if remote else cmd(args,timeout=15)
    d={"returncode":p.returncode,"stderr":p.stderr.strip()}
    for line in p.stdout.splitlines():
        if "=" in line:
            k,v=line.split("=",1); d[k]=v
    return d

def snapshot(remote=False):
    s=r"""date -Is
free -b
swapon --show --bytes || true
grep -E '^(pgfault|pgmajfault|pswpin|pswpout) ' /proc/vmstat || true
for f in /sys/class/drm/card*/device/mem_info_gtt_used /sys/class/drm/card*/device/gpu_busy_percent /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input /sys/class/net/thunderbolt0/statistics/rx_bytes /sys/class/net/thunderbolt0/statistics/tx_bytes /sys/class/net/thunderbolt0/statistics/rx_errors /sys/class/net/thunderbolt0/statistics/tx_errors; do if test -r "$f"; then printf '%s=' "$f"; cat "$f"; fi; done
powerprofilesctl get 2>/dev/null || true
"""
    p=ssh(["bash","-lc",s],timeout=30) if remote else cmd(["bash","-lc",s],timeout=30)
    return p.stdout+(("\nSTDERR:\n"+p.stderr) if p.stderr else "")

def stop_pair(lu,ru):
    lp=subprocess.Popen(["systemctl","--user","stop",lu+".service"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    rp=subprocess.Popen(SSH+["systemctl","--user","stop",ru+".service"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:lo,le=lp.communicate(timeout=60)
    except subprocess.TimeoutExpired: lp.kill();lo,le=lp.communicate()
    try:ro,re_=rp.communicate(timeout=60)
    except subprocess.TimeoutExpired: rp.kill();ro,re_=rp.communicate()
    active_local.discard(lu); active_remote.discard(ru)
    return {"local_stop":{"stdout":lo,"stderr":le},"remote_stop":{"stdout":ro,"stderr":re_},
            "local":status_unit(lu),"remote":status_unit(ru,True)}

def parse_csv(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))
def continuation_blob(log:Path,depth:int):
    text=log.read_text(errors="replace")
    marker=f'ds4-bench: gen[ctx={depth}] decoded text: '
    at=text.rfind(marker)
    if at<0:return None,None
    blob=text[at+len(marker):].strip()
    return hashlib.sha256(blob.encode()).hexdigest(),blob
def scan_faults(text):
    pats=[r"out of memory",r"gpu fault",r"segmentation",r"desynchron",r"\bnan\b",r"hip error"]
    return [p for p in pats if re.search(p,text,re.I)]

def run_case(label,depth,gen,port):
    arm=label[0]; rep=label[1:]
    cid=f"{depth}-{label}"
    cdir=RAW/"ab"/str(depth)/cid; cdir.mkdir(parents=True,exist_ok=True)
    term=cdir/"terminal.json"
    if term.exists():
        prior=json.loads(term.read_text())
        if prior.get("state")=="COMPLETE":return prior
    remote_dir=str(cdir); ssh(["mkdir","-p",remote_dir],timeout=20,check=True)
    lu=f"d4pg2-{depth}-{label.lower()}-c"; ru=f"d4pg2-{depth}-{label.lower()}-w"
    for remote,u in ((False,lu),(True,ru)):
        fn=ssh if remote else cmd
        fn(["systemctl","--user","stop",u+".service"],timeout=30)
        fn(["systemctl","--user","reset-failed",u+".service"],timeout=20)
    artifact=ART/arm
    csvp=cdir/"result.csv"; frontier=cdir/"frontiers"; frontier.mkdir(exist_ok=True)
    (cdir/"metrics.before.node01.txt").write_text(snapshot(False))
    (cdir/"metrics.before.node02.txt").write_text(snapshot(True))
    worker=["systemd-run","--user",f"--unit={ru}","--property=KillMode=control-group","--property=Restart=no",
            "--property=TimeoutStopSec=45",f"--property=StandardOutput=append:{remote_dir}/worker.log",
            f"--property=StandardError=append:{remote_dir}/worker.log",f"--setenv=LD_LIBRARY_PATH={LD}",
            "--setenv=OMP_NUM_THREADS=1","--setenv=DS4_TP_GATE_TIMEOUT_MS=5000",
            *clean_env(str(artifact/"ds4")),"--rocm","-m",str(MODEL),"--ctx",str(CTX),
            "--role","worker","--coordinator",COORD,str(port),"--tensor-parallel","--transport","tcp"]
    wp=ssh(worker,timeout=30)
    if wp.returncode: raise RuntimeError(f"{cid} worker launch: {wp.stderr or wp.stdout}")
    active_remote.add(ru); time.sleep(2)
    bench=[*clean_env(str(artifact/"ds4-bench-warm")),"--backend","rocm","-m",str(MODEL),
           "--prompt-file",str(CORPUS),"--ctx-start",str(depth),"--ctx-max",str(depth),"--ctx-alloc",str(CTX),
           "--gen-tokens",str(gen),"--show-output","--csv",str(csvp),
           "--dump-frontier-logits-dir",str(frontier),"--role","coordinator","--listen",COORD,str(port),
           "--tensor-parallel","--transport","tcp"]
    coord=["systemd-run","--user",f"--unit={lu}","--property=KillMode=control-group","--property=Restart=no",
           "--property=TimeoutStopSec=45",f"--property=StandardOutput=append:{cdir}/coordinator.log",
           f"--property=StandardError=append:{cdir}/coordinator.log",f"--setenv=LD_LIBRARY_PATH={LD}",
           "--setenv=OMP_NUM_THREADS=1","--setenv=DS4_TP_GATE_TIMEOUT_MS=5000",*bench]
    cp=cmd(coord,timeout=30)
    if cp.returncode:
        stop_pair(lu,ru); raise RuntimeError(f"{cid} coordinator launch: {cp.stderr or cp.stdout}")
    active_local.add(lu)
    record={"case_id":cid,"arm":arm,"rep":rep,"depth":depth,"gen":gen,"state":"IN_FLIGHT",
            "started_at":now(),"port":port,"bench_argv":bench}
    atomic(cdir/"registry.json",record)
    deadline=time.time()+(5400 if depth>=65536 else 3600)
    lastwrite=0
    while time.time()<deadline:
        if stop_requested: raise RuntimeError("stop requested")
        ls=status_unit(lu); rs=status_unit(ru,True)
        if time.time()-lastwrite>30:
            atomic(cdir/"live-status.json",{"updated_at":now(),"coordinator":ls,"worker":rs});lastwrite=time.time()
        if ls.get("ActiveState") in {"inactive","failed"}:break
        time.sleep(5)
    else: raise TimeoutError(f"{cid} deadline exceeded")
    ls=status_unit(lu); rs=status_unit(ru,True)
    stopped=stop_pair(lu,ru)
    rp=ssh(["cat",remote_dir+"/worker.log"],timeout=30)
    (cdir/"worker.remote.log").write_text(rp.stdout+(("\nSTDERR:\n"+rp.stderr) if rp.stderr else ""))
    (cdir/"metrics.after.node01.txt").write_text(snapshot(False))
    (cdir/"metrics.after.node02.txt").write_text(snapshot(True))
    rows=parse_csv(csvp) if csvp.exists() else []
    ff=frontier/f"frontier_{depth:06d}.logits.json"
    fsha=sha(ff) if ff.exists() else None
    csha,cblob=continuation_blob(cdir/"coordinator.log",depth)
    logs=(cdir/"coordinator.log").read_text(errors="replace")+(cdir/"worker.remote.log").read_text(errors="replace")
    faults=scan_faults(logs)
    execok=ls.get("Result") in {"success",""} and ls.get("ExecMainStatus")=="0"
    rowok=len(rows)==1 and rows[0].get("ctx_tokens")==str(depth) and rows[0].get("prefill_tokens")==str(depth)
    result={**record,"state":"COMPLETE","finished_at":now(),
            "status":"PASS_TECHNICAL" if execok and rowok and fsha and csha and not faults else "TECHNICAL_STOP",
            "coordinator":ls,"worker":rs,"pair_stop":stopped,"csv_rows":rows,
            "frontier_sha256":fsha,"continuation_sha256":csha,"continuation_blob":cblob,"faults":faults}
    atomic(term,result);return result

def analyze(cases,depth):
    rows=[c for c in cases if c["depth"]==depth]
    A=[c for c in rows if c["arm"]=="A"]; B=[c for c in rows if c["arm"]=="B"]
    def vals(cs,key): return [float(c["csv_rows"][0][key]) for c in cs]
    Ap=vals(A,"prefill_tps");Bp=vals(B,"prefill_tps");Ad=vals(A,"gen_tps");Bd=vals(B,"gen_tps")
    At=[depth/x for x in Ap];Bt=[depth/x for x in Bp]
    Amed=statistics.median(At);Bmed=statistics.median(Bt);Asd=statistics.pstdev(At)
    reduction=Amed-Bmed; reduction_pct=reduction/Amed*100
    faster=sum(t<Amed for t in Bt)
    technical=all(c.get("status")=="PASS_TECHNICAL" for c in rows)
    frontier_exact=len({c.get("frontier_sha256") for c in rows})==1
    continuation_exact=len({c.get("continuation_sha256") for c in rows})==1
    decode_guard=statistics.median(Bd)>=0.98*statistics.median(Ad)
    gate=technical and frontier_exact and continuation_exact and decode_guard and faster>=2 and reduction>max(Asd,0.01*Amed)
    return {"depth":depth,"A_prefill_tps":Ap,"B_prefill_tps":Bp,"A_decode_tps":Ad,"B_decode_tps":Bd,
            "A_prefill_s":At,"B_prefill_s":Bt,"A_median_s":Amed,"B_median_s":Bmed,
            "A_population_sd_s":Asd,"median_time_reduction_s":reduction,"median_time_reduction_pct":reduction_pct,
            "B_faster_than_A_median_count":faster,"technical_pass":technical,"frontier_exact":frontier_exact,
            "continuation_exact":continuation_exact,"decode_guard_pass":decode_guard,"gate_pass":gate,
            "goal_10pct_met":reduction_pct>=10}

def update(phase,cases,analysis=None):
    obj={"schema":"ds4-prefill-gap-002-ab-v1","campaign":CAMPAIGN,"state":"IN_FLIGHT","phase":phase,
         "updated_at":now(),"cases":cases}
    if analysis is not None:obj["analysis"]=analysis
    atomic(REGISTRY,obj);atomic(OWNER,{"campaign":CAMPAIGN,"state":"IN_FLIGHT","phase":phase,
                                      "pid":os.getpid(),"updated_at":obj["updated_at"],"raw":str(RAW)})

def restore(lockfp,reason):
    for u in list(active_local):
        try:cmd(["systemctl","--user","stop",u+".service"],timeout=45)
        except Exception:pass
    for u in list(active_remote):
        try:ssh(["systemctl","--user","stop",u+".service"],timeout=45)
        except Exception:pass
    try:fcntl.flock(lockfp.fileno(),fcntl.LOCK_UN)
    except Exception:pass
    out={"reason":reason,"started_at":now()}
    try:
        before=lifecycle();out["before"]={"http":before[0],"body":before[1]}
        if before[0]==200 and before[1].get("state")!="READY":
            st,b=lifecycle("POST","on",{"confirm":True});out["on"]={"http":st,"body":b}
        out["ready"]=wait_lifecycle("READY",1500);out["smoke"]=smoke("PREFILL-GAP-AB-RESTORE-OK")
        out["status"]="PASS" if out["smoke"]["pass"] else "FAIL"
    except Exception as e:out["status"]="FAIL";out["error"]=f"{type(e).__name__}: {e}"
    out["finished_at"]=now();atomic(RESTORE,out);return out

def sig(signum,frame):
    global stop_requested;stop_requested=True

def main():
    signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
    if TERMINAL.exists():
        print(TERMINAL.read_text(),flush=True);return
    pre=json.loads(PREREG.read_text());p1=json.loads(P1.read_text())
    if p1.get("status")!="PASS":raise RuntimeError("P1 is not PASS")
    for arm in ("A","B"):
        for name in ("ds4","ds4-bench-warm"):
            if not (ART/arm/name).exists():raise RuntimeError(f"missing {arm}/{name}")
    if not (RAW/"build-B/BUILD_COMPLETE").exists():raise RuntimeError("B build not complete")
    for arm in ("A","B"):
        local=sha(ART/arm/"ds4")
        p=ssh(["sha256sum",str(ART/arm/"ds4")],timeout=20)
        if p.returncode or p.stdout.split()[0]!=local:raise RuntimeError(f"NODE02 {arm} artifact mismatch")
    initial=lifecycle()
    if initial[0]!=200 or initial[1].get("state")!="READY":raise RuntimeError(f"E1 not READY: {initial}")
    atomic(REGISTRY,{"schema":"ds4-prefill-gap-002-ab-v1","campaign":CAMPAIGN,"state":"IN_FLIGHT",
                     "phase":"ACQUIRE","started_at":now(),"production_before":initial[1],"cases":[]})
    off=lifecycle("POST","off",{"confirm":True});offstate=wait_lifecycle("OFF",420)
    atomic(RAW/"gateway-off-before-ab.json",{"reply":off,"state":offstate})
    lockfp=LOCK.open("a+");fcntl.flock(lockfp.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=[];error=None;analysis={}
    try:
        port=19740
        for lab in ORDER16:
            res=run_case(lab,16384,512,port);cases.append(res);port+=1
            update("AB_16K",cases)
            if res["status"]!="PASS_TECHNICAL":raise RuntimeError(f"technical stop {res['case_id']}")
        analysis["16k"]=analyze(cases,16384);update("AB_16K_COMPLETE",cases,analysis)
        if analysis["16k"]["gate_pass"]:
            port=19800
            for lab in ORDER64:
                res=run_case(lab,65536,128,port);cases.append(res);port+=1
                update("AB_64K",cases,analysis)
                if res["status"]!="PASS_TECHNICAL":raise RuntimeError(f"technical stop {res['case_id']}")
            analysis["64k"]=analyze(cases,65536);update("AB_64K_COMPLETE",cases,analysis)
        else:
            analysis["64k"]={"status":"SKIPPED_GATE_16K_NOT_MET"}
            update("AB_64K_SKIPPED",cases,analysis)
    except Exception as e:error=f"{type(e).__name__}: {e}"
    finally:
        rest=restore(lockfp,error or "AB_COMPLETE");lockfp.close()
    terminal={"schema":"ds4-prefill-gap-002-ab-v1","campaign":CAMPAIGN,"state":"COMPLETE",
              "status":"PASS" if error is None and rest.get("status")=="PASS" else "PARTIAL_OR_STOP",
              "error":error,"cases":cases,"analysis":analysis,"restore":rest,"finished_at":now()}
    atomic(TERMINAL,terminal);atomic(REGISTRY,terminal)
    atomic(OWNER,{"campaign":CAMPAIGN,"state":"COMPLETE" if rest.get("status")=="PASS" else "RESTORE_FAILED",
                  "updated_at":terminal["finished_at"],"terminal":str(TERMINAL),"restore":str(RESTORE)})
    print(json.dumps({"status":terminal["status"],"error":error,"analysis":analysis,"restore":rest.get("status")},indent=2),flush=True)
if __name__=="__main__":main()
