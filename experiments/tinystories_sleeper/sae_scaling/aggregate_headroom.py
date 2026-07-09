#!/usr/bin/env python3
"""Pull all headroom_results/*.json from HF and build the per-config tables +
claim-checks for the FINDINGS doc. Run locally (only downloads small JSON)."""
import os, json
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
REPO = "dmanningcoe/sae-scaling-tinystories-sleeper"
api = HfApi(token=os.environ["HF_TOKEN"])
files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("headroom_results/")]
recs = []
for f in files:
    try:
        recs.append(json.load(open(hf_hub_download(REPO, f, repo_type="dataset", token=os.environ["HF_TOKEN"]))))
    except Exception as e:
        print("skip", f, e)

hyb = {r2: r2 for r2 in []}
hybrid = [r for r in recs if r.get("script") == "hybrid_sweep"]      # OV path (ln1)
resid  = [r for r in recs if r.get("script") == "resid_sweep_hybrid"] # conventional (resid_mid)
qkov   = [r for r in recs if r.get("script") == "qk_ov_grid"]
decomp = [r for r in recs if r.get("script") == "ov_qk_decomp"]

def key(r): return (r["width"], r["k"], r["seed"])

print("\n=== P1a OV cell (hybrid_sweep, ln1 SAE) ===")
print(f"{'width':>6} {'k':>3} {'seed':>4} {'OV_bare':>8} {'hybrid':>8} {'win':>5} {'n':>3}")
ovrows = {}
for r in sorted(hybrid, key=key):
    ovrows[key(r)] = (r["opt_ov"], r["opt_hybrid"])
    print(f"{r['width']:>6} {r['k']:>3} {r['seed']:>4} {str(r['opt_ov']):>8} {str(r['opt_hybrid']):>8} {r['winner']:>5} {r['n']:>3}")

print("\n=== P1b conventional cell (resid_sweep_hybrid, resid_mid SAE) ===")
print(f"{'width':>6} {'k':>3} {'seed':>4} {'CONV_bare':>9} {'hybrid':>8} {'win':>5} {'n':>3}")
convrows = {}
for r in sorted(resid, key=key):
    convrows[key(r)] = (r["opt_conv"], r["opt_hybrid"])
    print(f"{r['width']:>6} {r['k']:>3} {r['seed']:>4} {str(r['opt_conv']):>9} {str(r['opt_hybrid']):>8} {r['winner']:>5} {r['n']:>3}")

# ---- claim checks ----
def fnum(x): return x if isinstance(x, (int, float)) else None
print("\n=== CLAIM CHECKS ===")
# (a) hybrid << bare  (per cell)
a_ov = [(k, ovrows[k]) for k in ovrows if fnum(ovrows[k][0]) and fnum(ovrows[k][1])]
a_cv = [(k, convrows[k]) for k in convrows if fnum(convrows[k][0]) and fnum(convrows[k][1])]
ov_hold = sum(1 for k,(b,h) in a_ov if h < b)
cv_hold = sum(1 for k,(b,h) in a_cv if h < b)
print(f"(a) hybrid<bare: OV cell {ov_hold}/{len(a_ov)} configs; conv cell {cv_hold}/{len(a_cv)} configs")
# (c) OV bare < conventional bare  (matched config)
both = [k for k in ovrows if k in convrows and fnum(ovrows[k][0]) and fnum(convrows[k][0])]
c_hold = sum(1 for k in both if ovrows[k][0] < convrows[k][0])
print(f"(c) OV_bare < CONV_bare: {c_hold}/{len(both)} matched configs")
# (b) both cells' hybrids -> ~0.30
ovh = [ovrows[k][1] for k in ovrows if fnum(ovrows[k][1])]
cvh = [convrows[k][1] for k in convrows if fnum(convrows[k][1])]
import statistics as st
if ovh:  print(f"(b) OV hybrid: mean={st.mean(ovh):.3f} range=[{min(ovh):.3f},{max(ovh):.3f}] n={len(ovh)}")
if cvh:  print(f"(b) CONV hybrid: mean={st.mean(cvh):.3f} range=[{min(cvh):.3f},{max(cvh):.3f}] n={len(cvh)}")

print("\n=== P2 deployable qk+ov grid ===")
for r in sorted(qkov, key=key):
    print(f"d{r['width']} k{r['k']} s{r['seed']}: best={r['best']} ovf={r['ov_feature']} qkf={r['qk_feature']}")

print("\n=== P3 decomposition ===")
for r in decomp:
    print(f"pct_qk={r['pct_qk']} total={r['total']} t1(OV)={r['term1_ovvalue']} t2(QK)={r['term2_qkpattern']} t3(trig)={r['term3_trigger']} check={r['decomp_check']}")

# dump aggregate for the doc
out = {"ov": {f"{k[0]}_{k[1]}_{k[2]}": v for k,v in ovrows.items()},
       "conv": {f"{k[0]}_{k[1]}_{k[2]}": v for k,v in convrows.items()},
       "qkov": [{ "cfg": f"d{r['width']}_k{r['k']}_s{r['seed']}", "best": r["best"]} for r in qkov],
       "decomp": decomp, "n_recs": len(recs)}
json.dump(out, open("/tmp/headroom_agg.json", "w"), indent=1)
print(f"\n{len(recs)} result files; wrote /tmp/headroom_agg.json")
