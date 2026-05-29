"""Judge ONE finegrid cell, fully instrumented, writing to a known log path.
Usage: python judge_one_cell.py <cell>   e.g. qk_to_ov_finegrid/frarouting_qk_to_ov_ln1_gran1
"""
import os, sys, shutil, traceback
sys.path.insert(0, "/tmp")

LOG = "/tmp/judge_one_cell.log"
def log(*a):
    msg = " ".join(str(x) for x in a)
    with open(LOG, "a") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)

cell = sys.argv[1]
open(LOG, "w").close()
log(f"=== judge_one_cell {cell} ===")
try:
    import judge_loop_diff as jl
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    # sanity: does HF list the qualitative files for this cell?
    prefix = f"qwen14b/grid_diff/{cell}"
    files = [f for f in api.list_repo_files(jl.REPO, repo_type="dataset")
             if f.startswith(prefix + "/") and "/qualitative_" in f]
    log(f"qualitative files found on HF for cell: {len(files)}")
    for f in files:
        log("  ", f)
    spend = jl._load_spend()
    log("spend before:", round(spend["cumulative_usd"], 4))
    res = jl.process(api, cell, spend)
    log("process result:", res)
    log("spend after:", round(jl._load_spend()["cumulative_usd"], 4))
except Exception as e:
    log("EXCEPTION:", type(e).__name__, str(e))
    log(traceback.format_exc())
finally:
    for d in (f"/tmp/grid_diff/{cell}", f"/tmp/grid_diff/{cell}_hf",
              f"/tmp/grid_diff/{cell}_one"):
        shutil.rmtree(d, ignore_errors=True)
    log("=== done ===")
