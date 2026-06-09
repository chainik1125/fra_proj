# EXTRA_PIP: einops pyyaml
"""Overnight 5-seed sweep driver. Loops over a WORKLIST of 'config:seed' items, runs run_steer.py per
item (grid-exact 100k/12k SAE via its content-addressed cache), uploads each result with retry, and
SKIPS any item whose result already exists on HF with done:True -> fully resumable across pods/nights.

Per-item subprocess isolation: one item crashing (OOM, transient HF) does NOT kill the loop; the next
item proceeds, and a relaunch re-runs only the unfinished items. Few pods + retry-upload avoids the
HF commit-storm. Each pod takes a disjoint WORKLIST slice (env, partitioned by the launcher)."""
import os, sys, json, time, subprocess, pathlib
from huggingface_hub import hf_hub_download, upload_file
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data")
HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
TOK=os.environ.get("HF_TOKEN")
CODE=pathlib.Path(__file__).resolve().parent
WORK=[w.strip() for w in os.environ["WORKLIST"].split(",") if w.strip()]
PROG=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/seedsweep_progress.json")); PROG.parent.mkdir(parents=True,exist_ok=True)
prog={"worklist":WORK,"done":False,"n":len(WORK),"items":{}}
def save(done=False): prog["done"]=done; PROG.write_text(json.dumps(prog,indent=2))

def result_done(rel):
    try:
        lp=hf_hub_download(HF_REPO,rel,repo_type="dataset",local_dir="/tmp/ss_chk",token=TOK,force_download=True)
        return bool(json.load(open(lp)).get("done"))
    except Exception:
        return False

save()
print(f"[seedsweep] {len(WORK)} items: {WORK}",flush=True)
for i,item in enumerate(WORK):
    config,seed=item.rsplit(":",1)
    rname=f"{config}_seed{seed}_results.json"; rel=f"{HF_PREFIX}/results/{rname}"; outp=f"/workspace/out/{rname}"
    if result_done(rel):
        prog["items"][item]="skip(done)"; save(); print(f"[{i+1}/{len(WORK)}] SKIP {item} (already on HF)",flush=True); continue
    print(f"[{i+1}/{len(WORK)}] RUN {item}",flush=True); t0=time.time()
    # a union harvest that SIGBUS'd mid-write leaves a multi-GB acts_pool.dat that fills the disk and
    # poisons every later item; clear it (+ the on-demand dl cache) before each run as belt-and-suspenders.
    os.system("rm -f /workspace/acts_pool.dat /workspace/out/_sae_cache.pt; rm -rf /workspace/rs_dl /tmp/ss_chk")
    env={**os.environ,"CONFIG":config,"RUN_SEED":seed,"OUT_PATH":outp}
    r=subprocess.run([sys.executable,"run_steer.py"],env=env,cwd=str(CODE))
    if r.returncode!=0 or not pathlib.Path(outp).exists():
        prog["items"][item]=f"FAIL(rc={r.returncode})"; save(); print(f"[{i+1}/{len(WORK)}] FAIL {item} rc={r.returncode}",flush=True); continue
    up="dropped"
    for ua in range(8):   # retry: concurrent result commits can 429 the Hub
        try: upload_file(path_or_fileobj=outp,path_in_repo=rel,repo_id=HF_REPO,repo_type="dataset",token=TOK); up="ok"; break
        except Exception as e: print(f"[upload retry {ua}] {item}: {e}",flush=True); time.sleep(60)
    prog["items"][item]=f"done({up},{time.time()-t0:.0f}s)"; save()
    print(f"[{i+1}/{len(WORK)}] DONE {item} upload={up} ({time.time()-t0:.0f}s)",flush=True)
save(done=True); print("[seedsweep] worklist complete",flush=True)
