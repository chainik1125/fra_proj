#!/usr/bin/env python3
"""Combine the saved per-seed extracts (seed_extracts/seed{1,2,3,4,7}.json) into a mean+/-std table."""
import json, math, pathlib
SD=pathlib.Path(__file__).resolve().parent/"seed_extracts"; SEEDS=[1,2,3,4,7]
ex={s:json.load(open(SD/f"seed{s}.json"))["cells"] for s in SEEDS if (SD/f"seed{s}.json").exists()}
GROUPS=[("FRA",["grid_fra_base_ln1_L0","grid_fra_base_ln1_L1","grid_fra_base_ln1_L2","grid_fra_base_ln1_L3",
                "grid_fra_sleeper_ln1_L0","grid_fra_sleeper_ln1_L1","grid_fra_sleeper_ln1_L2","grid_fra_sleeper_ln1_L3"]),
        ("conv-sleeper",["grid_cs_sleeper_ln1_L1","grid_cs_sleeper_ln1_L2","grid_cs_sleeper_ln1_L3",
                "grid_cs_sleeper_rmid_L0","grid_cs_sleeper_rmid_L1","grid_cs_sleeper_rmid_L2","grid_cs_sleeper_rmid_L3",
                "grid_cs_sleeper_rpost_L0","grid_cs_sleeper_rpost_L1","grid_cs_sleeper_rpost_L2","grid_cs_sleeper_rpost_L3"]),
        ("conv-union",["grid_cs_union_ln1_L1","grid_cs_union_ln1_L2","grid_cs_union_ln1_L3",
                "grid_cs_union_rmid_L0","grid_cs_union_rmid_L1","grid_cs_union_rmid_L2","grid_cs_union_rmid_L3",
                "grid_cs_union_rpost_L0","grid_cs_union_rpost_L1","grid_cs_union_rpost_L2","grid_cs_union_rpost_L3"])]
def stat(xs):
    m=sum(xs)/len(xs); sd=math.sqrt(sum((x-m)**2 for x in xs)/len(xs)) if len(xs)>1 else 0.0; return m,sd
comb={}; best=[]
for gname,cfgs in GROUPS:
    print(f"\n[{gname}]   cell{'':22}  n   mean_J +/- std")
    for c in cfgs:
        js=[ex[s][c]["J"] for s in ex if c in ex[s]]
        short=c.replace("grid_","")
        if not js: print(f"   {short:28}  0    (pending)"); continue
        m,sd=stat(js); comb[c]={"n":len(js),"mean":round(m,4),"std":round(sd,4),"seeds":js}
        star=" *" if (len(js)>=4 and m<0.16) else ""
        print(f"   {short:28}  {len(js)}   {m:.3f} +/- {sd:.3f}{star}")
        if len(js)>=4: best.append((c,m,sd,len(js)))
(SD/"combined.json").write_text(json.dumps(comb,indent=2))
print("\n=== best cells (mean J, n>=4) ===")
for c,m,sd,n in sorted(best,key=lambda r:r[1])[:8]:
    print(f"   {c.replace('grid_',''):28}  {m:.3f} +/- {sd:.3f}  (n={n})")
print(f"\nsaved -> seed_extracts/combined.json")
