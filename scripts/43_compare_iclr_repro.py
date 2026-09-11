"""Compare our reproduction of Dmitry's ICLR summary against his committed numbers.

His 14 settings live in docs/dmitry/INDUCTIVE_BACKDOOR_MAP.md on upstream/iclr-summary;
docs/dmitry/inductive_backdoor_figures/build.py names the data file behind each figure.
`results/iclr_repro/targets.json` holds his values, extracted with build.py's own maths.
`results/iclr_repro/ours/` holds ours, produced by running HIS scripts unmodified from a
checkout of iclr-summary at 292b643, under an env pinned to HIS pod versions
(sae_lens 5.10.7, transformer_lens 2.18.0) on an NCSA H100.

Settings 2,3,5,6,7,8 were values he typed into build.py from logs rather than JSON, so our
side for those is transcribed from our run logs (kept in results/iclr_repro/logs_summary.md).

Run: python scripts/43_compare_iclr_repro.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path("results/iclr_repro")
T = json.loads((R / "targets.json").read_text())
O = R / "ours"
j = lambda p: json.loads((O / p).read_text())
rows: list[tuple] = []


def add(setting, quantity, his, ours, tol=0.02, floor=0.005):
    # his log-reported values are rounded to 2-3 dp, so allow an absolute floor as well
    ok = "match" if abs(his - ours) <= max(tol * max(abs(his), abs(ours)), floor) else "DIFFERS"
    rows.append((setting, quantity, his, ours, ok))


# 1 Gemma word-association: mean collateral at 30% suppression
d = j("S01_g4_sc/g4.json")["rows"]
for m, lab in (("fra", "FRA-QK"), ("dom", "DoM"), ("conv", "12-feat SAE")):
    ours = float(np.mean([np.interp(.3, *zip(*sorted(r[m]))) for r in d]))
    add("1 Gemma word-assoc", f"collateral@30% {lab}", T["S01_gemma_collateral_at_30pct"][m], ours)

# 2 GPT-2 (log-reported)
for k, v in zip(("fra", "dom", "sae12", "payload"), (0.068, 1.829, 6.055, 4.119)):
    add("2 GPT-2 word-assoc", f"collateral@80% {k}", T["S02_gpt2_collateral_at_80pct"][k], v)

# 3 many-shot (log-reported)
add("3 many-shot", "P(marker) FRA", T["S03_manyshot_P_marker"]["fra"], 0.687)
add("3 many-shot", "P(marker) DoM", T["S03_manyshot_P_marker"]["dom"], 0.0)

# 4 instruction injection: curve endpoints
c = j("S04_injsel_sc/selectivity.json")["curves"]
add("4 instruction inj", "FRA max removal", max(r[0] for r in T["S04_injection_curves"]["fra"]),
    max(r["removal"] for r in c["fra"]))
add("4 instruction inj", "FRA hard-retention", T["S04_injection_curves"]["fra"][0][1], c["fra"][0]["hard"])

# 5 retrieval (log-reported)
add("5 box retrieval", "legit KL FRA", T["S05_retrieval_KL"]["fra"], 0.0016)
add("5 box retrieval", "legit KL steer", T["S05_retrieval_KL"]["steer"], 1.7733)
add("5 box retrieval", "P(frog) FRA", T["S05_retrieval_KL"]["P_frog_fra"], 0.042)

# 6,7,8 (log-reported)
add("6 shared payload", "P(frog) target", T["S06_shared_P_frog"]["fra_diff"][0], 0.131)
add("6 shared payload", "P(frog) sibling", T["S06_shared_P_frog"]["fra_diff"][1], 0.083)
add("7 entity sibling", "P target (FRA)", T["S07_entity_P"]["fra_diff"][0], 0.758)
add("7 entity sibling", "P sibling (FRA)", T["S07_entity_P"]["fra_diff"][1], 0.366)
for i, v in enumerate((0.606, 0.522, 0.051, 0.060)):
    add("8 digit-class", f"P union-cut case{i+1}", T["S08_class_P"]["fra_union"][i], v)

# 9 binding
# the summary file is the FINAL curve (8 points incl. the c=1e9 mask oracle);
# binding_selectivity.json is an intermediate checkpoint with only 7.
b = j("S09_bind_sc/binding_selectivity_summary.json")
add("9 binding", "mask-oracle target supp", T["S09_binding"]["mask_oracle"][0], b["fra_curve"][-1]["target_supp"])
add("9 binding", "mask-oracle sibling coll", T["S09_binding"]["mask_oracle"][1], b["fra_curve"][-1]["sib_coll"])
add("9 binding", "FRA point1 target supp", T["S09_binding"]["fra"][0][0],
    [r for r in b["fra_curve"] if r["c"] < 1e8][0]["target_supp"])

# 10 factual editing
f = j("S10_fact_sc/complete_summary.json")
add("10 factual editing", "median best drop", T["S10_facts"]["target"], f["median_best_drop"])
add("10 factual editing", "median collateral", T["S10_facts"]["sibling"], f["median_collateral_best"])

# 11 persistence
p = j("S11_persist_sc/persist_results.json")["pooled"]
for k in ("rem_holdout", "rem_holdout_k3", "rem_random_FRA", "oracle_ceiling"):
    add("11 persistence", k, T["S11_persistence"][k], p[k])

# 12 weight-sparse
w = j("S12_wsb_sc/backdoor_results.json")["models"]
for m in ("sparse", "dense"):
    for k in ("median_win_fradiff_vs_payload_mask", "median_win_fradiff_vs_payload_suppress"):
        add("12 weight-sparse", f"{m} {k.split('_vs_')[1]}", T["S12_sparse_ratio"][m][k], w[m]["summary"][k])

# 14 SSN
fr = max(j("S14_fra5_sc/pii_sweep_fra5-svd.json")["rows"], key=lambda r: r["emit_supp"])
sa = next(r for r in j("S14_sae_sc/pii_sweep_sae-5.json")["rows"]
          if r["L"] == 18 and r["k"] == 1 and r["strength"] == 1)
add("14 SSN disclosure", "P(digit) best FRA", T["S14_ssn_P_first_digit"]["fra_best"], fr["emit_p"])
add("14 SSN disclosure", "P(digit) SAE", T["S14_ssn_P_first_digit"]["sae"], sa["emit_p"])

w1, w2 = max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)
out = [f"| {'setting':{w1}} | {'quantity':{w2}} | {'his':>10} | {'ours':>10} | verdict |",
       f"|{'-'*(w1+2)}|{'-'*(w2+2)}|{'-'*12}|{'-'*12}|---------|"]
for s, q, h, o, ok in rows:
    out.append(f"| {s:{w1}} | {q:{w2}} | {h:10.4f} | {o:10.4f} | {ok} |")
n_ok = sum(1 for r in rows if r[4] == "match")
out.append(f"\n{n_ok}/{len(rows)} quantities match within 2%.")
txt = "\n".join(out)
print(txt)
(R / "comparison.md").write_text(txt + "\n", encoding="utf-8")
