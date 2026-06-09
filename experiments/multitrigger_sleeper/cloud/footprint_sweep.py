# EXTRA_PIP: einops pyyaml
"""Footprint sweep for the SAE methods (FRA-OV + conv-SAE): hold feature-selection fixed (seed-7 cached
SAE, same cell), vary ONLY the injection footprint ∈ {trigger,prompt,all,rollout} via the FOOTPRINT env
override. Isolates the footprint effect for the SAE methods the way the mean-diff fpgrid did for DoM/CAA.
WORKLIST items are 'config:footprint'. Resumable (skips done), retry-upload."""
import os, sys, json, time, subprocess, pathlib
from huggingface_hub import hf_hub_download, upload_file
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data"); HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
TOK=os.environ.get("HF_TOKEN"); CODE=pathlib.Path(__file__).resolve().parent
WORK=[w.strip() for w in os.environ["WORKLIST"].split(",") if w.strip()]
PROG=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/fpsweep_progress.json")); PROG.parent.mkdir(parents=True,exist_ok=True)
prog={"worklist":WORK,"done":False,"items":{}}
def save(d=False): prog["done"]=d; PROG.write_text(json.dumps(prog,indent=2))
def result_done(rel):
    try: return bool(json.load(open(hf_hub_download(HF_REPO,rel,repo_type="dataset",local_dir="/tmp/fp_chk",token=TOK,force_download=True))).get("done"))
    except Exception: return False
save(); print(f"[fpsweep] {len(WORK)} items: {WORK}",flush=True)
for i,item in enumerate(WORK):
    parts=item.split(":")                       # config:footprint[:seed]  (seed defaults to 7)
    config,fp=parts[0],parts[1]; seed=parts[2] if len(parts)>2 else "7"
    rname=f"{config}_fp{fp}_seed{seed}_results.json"; rel=f"{HF_PREFIX}/results/{rname}"; outp=f"/workspace/out/{rname}"
    if result_done(rel):
        prog["items"][item]="skip(done)"; save(); print(f"[{i+1}/{len(WORK)}] SKIP {item}",flush=True); continue
    print(f"[{i+1}/{len(WORK)}] RUN {item}",flush=True); t0=time.time()
    os.system("rm -f /workspace/acts_pool.dat; rm -rf /workspace/rs_dl /tmp/fp_chk")
    env={**os.environ,"CONFIG":config,"RUN_SEED":seed,"FOOTPRINT":fp,"OUT_PATH":outp}
    r=subprocess.run([sys.executable,"run_steer.py"],env=env,cwd=str(CODE))
    if r.returncode!=0 or not pathlib.Path(outp).exists():
        prog["items"][item]=f"FAIL(rc={r.returncode})"; save(); print(f"[{i+1}/{len(WORK)}] FAIL {item}",flush=True); continue
    up="dropped"
    for ua in range(8):
        try: upload_file(path_or_fileobj=outp,path_in_repo=rel,repo_id=HF_REPO,repo_type="dataset",token=TOK); up="ok"; break
        except Exception as e: print(f"[upload retry {ua}] {e}",flush=True); time.sleep(60)
    prog["items"][item]=f"done({up},{time.time()-t0:.0f}s)"; save(); print(f"[{i+1}/{len(WORK)}] DONE {item} ({time.time()-t0:.0f}s)",flush=True)
save(True); print("[fpsweep] complete",flush=True)
