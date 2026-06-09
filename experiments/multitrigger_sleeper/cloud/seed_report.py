#!/usr/bin/env python3
"""Per-seed extraction (one seed at a time). For env SEED, pull each cell's result, extract the best
(ASR<=0.05, min J_clean) over coeff_sweep + topk_sweep_c1 + full_grid, print a grouped table, and SAVE
a durable record to seed_extracts/seed<SEED>.json. Seed 7 = the original grid (<config>_results.json);
seeds 1-4 = the sweep (<config>_seed<N>_results.json)."""
import os, json, pathlib
from huggingface_hub import hf_hub_download
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat/results"; TOK=os.environ.get("HF_TOKEN")
SEED=os.environ["SEED"]
GROUPS=[("FRA",["grid_fra_base_ln1_L0","grid_fra_base_ln1_L1","grid_fra_base_ln1_L2","grid_fra_base_ln1_L3",
                "grid_fra_sleeper_ln1_L0","grid_fra_sleeper_ln1_L1","grid_fra_sleeper_ln1_L2","grid_fra_sleeper_ln1_L3"]),
        ("conv-sleeper",["grid_cs_sleeper_ln1_L1","grid_cs_sleeper_ln1_L2","grid_cs_sleeper_ln1_L3",
                "grid_cs_sleeper_rmid_L0","grid_cs_sleeper_rmid_L1","grid_cs_sleeper_rmid_L2","grid_cs_sleeper_rmid_L3",
                "grid_cs_sleeper_rpost_L0","grid_cs_sleeper_rpost_L1","grid_cs_sleeper_rpost_L2","grid_cs_sleeper_rpost_L3"]),
        ("conv-union",["grid_cs_union_ln1_L1","grid_cs_union_ln1_L2","grid_cs_union_ln1_L3",
                "grid_cs_union_rmid_L0","grid_cs_union_rmid_L1","grid_cs_union_rmid_L2","grid_cs_union_rmid_L3",
                "grid_cs_union_rpost_L0","grid_cs_union_rpost_L1","grid_cs_union_rpost_L2","grid_cs_union_rpost_L3"])]

def fname(c): return f"{c}_results.json" if SEED=="7" else f"{c}_seed{SEED}_results.json"
def fetch(c):
    try: return json.load(open(hf_hub_download(HF_REPO,f"{PFX}/{fname(c)}",repo_type="dataset",local_dir="/tmp/sr_dl",token=TOK,force_download=True)))
    except Exception: return None
def best(res):
    if not res or not res.get("done"): return None
    pts=[]
    for k in ("coeff_sweep","topk_sweep_c1","full_grid"):
        v=res.get(k)
        if isinstance(v,list): pts+=[p for p in v if isinstance(p,dict) and "Jclean" in p]
    if not pts: return None
    ok=[p for p in pts if p.get("ASR",1)<=0.05]
    b=min(ok,key=lambda p:p["Jclean"]) if ok else min(pts,key=lambda p:(p.get("ASR",1),p["Jclean"]))
    at="".join(s for s in (f"K{b['K']}" if 'K' in b else "", f"c{b['c']}" if 'c' in b else "") if s) or "-"
    return {"J":round(b["Jclean"],4),"ASR":round(b.get("ASR",0),4),"at":at,"removed":bool(ok)}

out={"seed":SEED,"cells":{}}
print(f"\n================  SEED {SEED}  ================")
for gname,cfgs in GROUPS:
    print(f"\n[{gname}]   cell{'':24}  best_J   ASR    @")
    for c in cfgs:
        b=best(fetch(c)); short=c.replace("grid_","")
        if b is None: print(f"   {short:30}   --       --     (pending)"); continue
        flag="" if b["removed"] else "  (no removal: ASR>0.05)"
        out["cells"][c]=b
        print(f"   {short:30}  {b['J']:.3f}   {b['ASR']:.2f}   {b['at']}{flag}")
sd=pathlib.Path(__file__).resolve().parent/"seed_extracts"; sd.mkdir(exist_ok=True)
(sd/f"seed{SEED}.json").write_text(json.dumps(out,indent=2))
print(f"\nsaved {len(out['cells'])} cells -> seed_extracts/seed{SEED}.json")
