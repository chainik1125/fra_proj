"""Rebuild the opening-list figures from saved results; no model runs.

Values transcribed from logs are identified beside their use. Keep these plots
descriptive: no pooled advantage ratios across different tasks or metrics.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fra-writeup-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BLUE, ORANGE, GREY, PURPLE = "#2864b4", "#c76b28", "#89939d", "#8262a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.labelcolor": "#303a45",
    "text.color": "#202b36", "axes.edgecolor": "#b6bec7",
    "savefig.facecolor": "white"})

def read(path):
    return json.loads((ROOT / "experiments" / path).read_text())

def canvas(title, ylabel, note=""):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=.12, right=.97, bottom=.23, top=.84)
    fig.suptitle(title, x=.12, ha="left", y=.97, fontsize=15, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.12, .025, note, fontsize=9, color="#5c6874", va="bottom")
    return fig, ax

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)

def bars(name, title, labels, values, ylabel, note="", colors=None, percent=False):
    fig, ax = canvas(title, ylabel, note)
    rects = ax.bar(labels, values, color=colors or [BLUE]*len(values), width=.58)
    ax.bar_label(rects, labels=[f"{v:.1%}" if percent else f"{v:.3g}" for v in values], padding=5)
    ax.set_ylim(0, 1.15 if percent else max(values)*1.22)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, name)

def grouped(name, title, labels, series, ylabel, note="", percent=False):
    fig, ax = canvas(title, ylabel, note)
    x = np.arange(len(labels)); w = .75 / len(series)
    for j, (label, vals, color) in enumerate(series):
        r = ax.bar(x + (j-(len(series)-1)/2)*w, vals, w*.92, label=label, color=color)
        ax.bar_label(r, labels=[f"{v:.1%}" if percent else f"{v:.3g}" for v in vals], padding=4, fontsize=9)
    ax.set_xticks(x, labels)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncol=len(series), frameon=False, fontsize=10)
    ax.set_ylim(0, 1.12 if percent else max(max(s[1]) for s in series)*1.25)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, name)

# 1. Poster g4 data: common support, with per-pair reach disclosed.
d = read("fra_win/out/g4/g4.json")["rows"]
fig, ax = canvas("Gemma-2: collateral at 30% probability suppression",
                 "Held-out summed KL (nats; lower is better)",
                 "Same four cases for every method; coarse-curve interpolation.\nDoM and 12-feature SAE fixed at L6; FRA uses several layers.")
x = np.arange(4); w = .23
for j, (m, label, color) in enumerate([("fra", "FRA-QK", BLUE), ("dom", "DoM", ORANGE), ("conv", "12-feature SAE", PURPLE)]):
    vals=[]
    for r in d:
        curve=sorted(r[m]); a,b=zip(*curve)
        assert min(a) <= .3 <= max(a)
        vals.append(np.interp(.3,a,b))
    rr=ax.bar(x+(j-1)*w,vals,w*.94,label=label,color=color)
    ax.bar_label(rr,fmt="%.2f",padding=4,fontsize=9)
ax.set_xticks(x,[f"{r['T'].strip()} → {r['P'].strip()}\nFRA maximum: {max(a for a,b in r['fra']):.1%}" for r in d])
ax.set_yscale("log"); ax.set_ylim(.1,65)
ax.legend(loc="upper left",bbox_to_anchor=(0,1.18),ncol=3,frameon=False,fontsize=10)
save(fig,"01_gemma")

# 2–3. Reported means/representative run from fra_win/INCONTEXT_LOG.md IC4 and jb3.
bars("02_gpt2", "GPT-2: collateral at 80% probability suppression",
     ["FRA-QK", "DoM", "12-feature\nSAE", "Payload\nsuppression"], [.07,1.83,6.06,4.12],
     "Held-out summed KL (nats; lower is better)",
     "Reported means over four associations; source: INCONTEXT_LOG.md, IC4.",
     [BLUE,ORANGE,PURPLE,GREY])
bars("03_manyshot", "Many-shot injection: probability of the planted marker",
     ["No intervention", "FRA: full\ninduction chain", "DoM"], [.80,.69,0.0],
     "P(Absolutely) on injected prompt", "Approximate values from INCONTEXT_LOG.md, jb3; PT SAEs on an IT model.\nDoM also suppressed legitimate marker use (0.29 → approximately 0).",
     [GREY,BLUE,ORANGE], True)

# 4. Full measured curves instead of an off-curve 85% comparison.
d=read("fra_organisms/results/injection_selectivity_NULL.json")
fig,ax=canvas("Single instruction injection: removal versus retention",
              "Hard legitimate-instruction retention",
              "60 injected prompts; 17 hard controls; saved sweep points joined in scale order.\nDoM L12 is the selected layer; PT SAEs on an IT model.")
for label,rows,color in [("FRA-QK",d['curves']['fra'],BLUE),("DoM, L12",[r for r in d['curves']['lin'] if r['layer']==12],ORANGE)]:
    ax.plot([r['removal'] for r in rows],[r['hard'] for r in rows],"o-",label=label,color=color)
ax.set_xlabel("Injection removal"); ax.set_xlim(0,1.03); ax.set_ylim(0,1.08)
ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.legend(frameon=False,loc="lower left")
save(fig,"04_instruction")

# 5. CAMPAIGN_REPORT.md §7: operating-point probabilities differ slightly.
bars("05_retrieval", "Simple box retrieval: preserving legitimate ‘frog’ uses",
     ["FRA-QK", "Content-gated\nprojection steer"], [.0016,1.77],
     "Legitimate-context KL (nats; lower is better)",
     "CAMPAIGN_REPORT.md §7; target P(frog): 0.213 → 0.042 (FRA), 0.024 (steer).\nSingle selected association; near-matched rather than identical removal.", [BLUE,ORANGE])

# 6. CAMPAIGN_REPORT Sprint 2 + out/shared_endpoint_diff.log.
grouped("06_shared", "Shared payload: specificity trades off against removal",
        ["Target: red box", "Sibling: blue box"],
        [("No intervention",[.199,.095],GREY),("FRA, ordinary pairs",[.083,.046],BLUE),
         ("FRA, differential pairs",[.134,.083],PURPLE)], "P(frog)",
        "Both boxes contain frog; target probability should fall while sibling probability stays high.\nSources: CAMPAIGN_REPORT.md, Sprint 2; shared_endpoint_diff.log.")

# 7. out/pii_sibling.log; the original WIN label was rejected for zero target effect.
grouped("07_entity", "Entity retrieval: the FRA cut leaves the target intact",
        ["Target entity", "Sibling entity"],
        [("No intervention",[.756,.388],GREY),("FRA, differential pairs",[.757,.364],BLUE),
         ("Content-gated steer",[.013,.001],ORANGE)], "P(retrieved value)",
        "Source: pii_sibling.log; FRA has no on-target suppression, so its small collateral is not a win.")

# 8. out/class_union.log, all four saved cases.
grouped("08_class", "Digit-class union: no suppression of digit retrieval",
        ["red → 7", "green → 3", "blue → frog", "gold → lamp"],
        [("No intervention",[.569,.490,.060,.056],GREY),("FRA class union",[.606,.521,.051,.060],BLUE)],
        "P(target value)", "Source: class_union.log; first two cases are intended targets, last two are word controls.")

# 9. Separate real feature edits from the attention-mask oracle explicitly.
d=read("fra_organisms2/results/binding_selectivity_WIN.json")
fig,ax=canvas("Variable binding: feature edits versus the mask oracle",
              "Sibling probability suppression (lower is better)",
              "Source: binding_selectivity_WIN.json, interpreted with binding_redteam.md.\nThe mask point is not an FRA feature edit; DoM points with total degeneration omitted.")
fr=[r for r in d['fra_curve'] if r['c']<1e8]; mask=d['fra_curve'][-1]
lin=[r for r in d['lin_curve_best_layer'] if r['degen']<1]
ax.plot([r['target_supp'] for r in fr],[r['sib_coll'] for r in fr],"o-",color=BLUE,label="FRA feature edits")
ax.scatter([mask['target_supp']],[mask['sib_coll']],marker="*",s=170,color=GREY,label="Attention-mask oracle")
ax.scatter([r['target_supp'] for r in lin],[r['sib_coll'] for r in lin],color=ORANGE,s=70,label="DoM, noncollapsed point")
ax.set_xlabel("Target probability suppression"); ax.set_xlim(0,1); ax.set_ylim(-.025,1.05)
ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.legend(frameon=False,loc="upper right",fontsize=10)
save(fig,"09_binding")

# 10. Final factual-editing summary, one common operating point.
d=read("fra_organisms/results/factedit_complete_NOGO_summary.json")
bars("10_facts", "Factual editing: target suppression spills onto siblings",
     ["Target fact", "Sibling facts"], [d['median_best_drop'],d['median_collateral_best']],
     "Median probability suppression", "20 facts; best FRA setting per fact; source: factedit_complete_NOGO_summary.json.\nDesired: strong target suppression and weak sibling suppression.", [BLUE,ORANGE], True)

# 11. Final persistence run, not the earlier single-pair summaries.
p=read("fra_persistence/results/persist_results_FINAL.json")['pooled']
bars("11_persistence", "Persistence: removal on held-out contexts and positions",
     ["FRA:\none cell", "FRA:\nthree cells", "FRA:\ndiagnosed union", "Embedding\ncut", "Oracle\nceiling"],
     [p['rem_holdout'],p['rem_holdout_k3'],p['rem_random_FRA'],p['rem_random_embedding'],p['oracle_ceiling']],
     "Probability suppression", "Three associations, 101 appearances; source: persist_results_FINAL.json.",
     [BLUE,BLUE,BLUE,ORANGE,GREY], True)

# 12. Median of pairwise ratios at each pair's matched point; exclude zero-target wsda.
d=read("fra_ws_backdoor/results/backdoor_results.json")['models']
fig,ax=canvas("Sparse-code backdoors: collateral advantage fails to transfer",
              "Baseline / FRA collateral ratio", "Differential FRA cuts; medians of pairwise matched-removal ratios across three associations.\nAbove 1 favors FRA; zero-removal wsda comparison omitted.")
x=np.arange(2); w=.32
for j,(key,label,color) in enumerate([('median_win_fradiff_vs_payload_mask','Payload mask',ORANGE),('median_win_fradiff_vs_payload_suppress','Payload suppression',PURPLE)]):
    vals=[d[m]['summary'][key] for m in ['sparse','dense']]
    rr=ax.bar(x+(j-.5)*w,vals,w*.94,label=label,color=color)
    ax.bar_label(rr,fmt="%.2f×",padding=4)
ax.axhline(1,color=GREY,linestyle='--'); ax.set_ylim(0,1.25)
ax.set_xticks(x,['Sparse substrate','Dense substrate']); ax.legend(frameon=False,loc='upper right',fontsize=10)
save(fig,"12_sparse")

# 13. Plot observed reach only: the frontier changes DoM gating on clean controls.
a=read("constrained_belief_updating/bridge_to_real/out/bridge3_results.json")
b=read("constrained_belief_updating/bridge_to_real/out/bridge3_replicate.json")
grouped("13_trained", "Trained backdoors: FRA removal varies sharply by seed",
        ["Fixed payload\nseed 0", "Context copy\nseed 0", "Fixed payload\nseed 2", "Context copy\nseed 2"],
        [("FRA-QK",[a['carrier_A']['qk_max'],a['carrier_B']['qk_max'],b['carrier_A_s2']['qk_max'],b['carrier_B_s2']['qk_max']],BLUE),
         ("DoM",[a['carrier_A']['dom_max'],a['carrier_B']['dom_max'],b['carrier_A_s2']['dom_max'],b['carrier_B_s2']['dom_max']],ORANGE)],
        "Maximum probability suppression", "One evaluation sequence per variant and seed; removal only, not a collateral comparison.\nSource: bridge3_results.json and bridge3_replicate.json.", True)

# 14. Best saved low-rank FRA (rank 2), not the summary's erroneous rank-1 label.
d=read("fra_pii/results/pii_sweep_fra5-svd.json")
fr=max(d['rows'],key=lambda r:r['emit_supp'])
sr=next(r for r in read("fra_pii/results/pii_sweep_sae-5.json")['rows'] if r['L']==18 and r['k']==1 and r['strength']==1)
bars("14_ssn", "SSN disclosure: lower token probability is not disarming",
     ["No intervention\nSSN emitted", "Best low-rank FRA\nSSN still emitted", "Single SAE feature\nSSN not emitted"],
     [d['base_emit'],fr['emit_p'],sr['emit_p']], "P(first SSN digit)",
     "One fixed database; generation outcomes from saved runs; SAE edit restricted to digit positions.\nFRA uses rank 2 here; sources: pii_sweep_fra5-svd.json and pii_sweep_sae-5.json.", [GREY,BLUE,ORANGE], True)

print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
