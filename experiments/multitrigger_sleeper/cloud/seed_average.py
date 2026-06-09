#!/usr/bin/env python3
"""Seed-average the conv+FRA grid: for each of the 30 cells, pull results for seeds {1,2,3,4} (sweep
driver, named <config>_seed<N>_results.json) + the existing seed-7 grid run (<config>_results.json),
extract best (ASR<=0.05, min J_clean) per seed, report mean +/- std over the 5 seeds. Run LOCALLY
after the overnight sweep completes (reads from HF). Usage: HF_TOKEN=... python3 seed_average.py"""
import os, json, math
from huggingface_hub import hf_hub_download
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat/results"; TOK=os.environ.get("HF_TOKEN")
SEEDS=[1,2,3,4,7]
CONFIGS=["grid_fra_base_ln1_L0","grid_fra_base_ln1_L1","grid_fra_base_ln1_L2","grid_fra_base_ln1_L3",
 "grid_fra_sleeper_ln1_L0","grid_fra_sleeper_ln1_L1","grid_fra_sleeper_ln1_L2","grid_fra_sleeper_ln1_L3",
 "grid_cs_sleeper_ln1_L1","grid_cs_sleeper_ln1_L2","grid_cs_sleeper_ln1_L3",
 "grid_cs_sleeper_rmid_L0","grid_cs_sleeper_rmid_L1","grid_cs_sleeper_rmid_L2","grid_cs_sleeper_rmid_L3",
 "grid_cs_sleeper_rpost_L0","grid_cs_sleeper_rpost_L1","grid_cs_sleeper_rpost_L2","grid_cs_sleeper_rpost_L3",
 "grid_cs_union_ln1_L1","grid_cs_union_ln1_L2","grid_cs_union_ln1_L3",
 "grid_cs_union_rmid_L0","grid_cs_union_rmid_L1","grid_cs_union_rmid_L2","grid_cs_union_rmid_L3",
 "grid_cs_union_rpost_L0","grid_cs_union_rpost_L1","grid_cs_union_rpost_L2","grid_cs_union_rpost_L3"]

def fetch(name):
    try: return json.load(open(hf_hub_download(HF_REPO,f"{PFX}/{name}",repo_type="dataset",local_dir="/tmp/sa_dl",token=TOK,force_download=True)))
    except Exception: return None

def best_J(res):
    """min J over all sweep points with ASR<=0.05; return (J, ASR) or None."""
    if not res or not res.get("done"): return None
    pts=[]
    for key in ("coeff_sweep","topk_sweep","grid","results"):
        v=res.get(key)
        if isinstance(v,list): pts+=[p for p in v if isinstance(p,dict) and "Jclean" in p]
        elif isinstance(v,dict):
            for vv in v.values():
                if isinstance(vv,list): pts+=[p for p in vv if isinstance(p,dict) and "Jclean" in p]
    ok=[p for p in pts if p.get("ASR",1)<=0.05]
    if not ok: return None
    b=min(ok,key=lambda p:p["Jclean"]); return (b["Jclean"], b.get("ASR",0))

def stats(xs):
    m=sum(xs)/len(xs); sd=math.sqrt(sum((x-m)**2 for x in xs)/len(xs)) if len(xs)>1 else 0.0; return m,sd

print(f"{'config':32} {'n':>2}  {'J mean':>7} {'J std':>6}  per-seed J")
rows=[]
for c in CONFIGS:
    js={}
    for s in SEEDS:
        nm=f"{c}_results.json" if s==7 else f"{c}_seed{s}_results.json"
        bj=best_J(fetch(nm))
        if bj is not None: js[s]=bj[0]
    if not js: print(f"{c:32} {0:>2}  {'--':>7}"); continue
    m,sd=stats(list(js.values()))
    rows.append((c,len(js),m,sd))
    print(f"{c:32} {len(js):>2}  {m:>7.3f} {sd:>6.3f}  "+" ".join(f"s{s}={js[s]:.3f}" for s in SEEDS if s in js))
print("\n=== best cells (lowest seed-mean J, n>=3) ===")
for c,n,m,sd in sorted([r for r in rows if r[1]>=3],key=lambda r:r[2])[:8]:
    print(f"  {c:32} J={m:.3f} +/- {sd:.3f} (n={n})")
