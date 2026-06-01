#!/usr/bin/env python3
"""F603 focused cross-finetune analysis (companion to CROSS_FINETUNE_SUMMARY.md §9).

Read-only on HF (no GPU). Produces, for feature F603 across the three Qwen-14B EM
finetunes (financial / medical / sports):
  1. steering effect Δalign@50 + per-cell rank BY STEERING EFFECT  (main grids)
  2. rank BY ATTRIBUTION SCORE                                     (ranking JSONs)
  3. the base-model control: F603 steered on base vs EM            (f603/qk finegrids)
  4. two figures: F603_steering_curves.png, F603_base_vs_em.png

Usage:  HF_TOKEN=... python experiments/fra_14b_diff/f603_analysis.py
"""
import os, re, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import grid_metrics as gm

TOK = os.environ["HF_TOKEN"]; REPO = "dmanningcoe/fra-phase1-steering-data"
FEAT = 603; OUTDIR = os.path.dirname(os.path.abspath(__file__))
api = HfApi(token=TOK); ALL = api.list_repo_files(REPO, repo_type="dataset")
dl = lambda f: json.loads(open(hf_hub_download(REPO, f, repo_type="dataset", token=TOK)).read())
CAMPS = [("financial", "qwen14b/grid_diff", "finance"),
         ("medical", "qwen14b/grid_diff_medical", "medical"),
         ("sports", "qwen14b/grid_diff_sports", "sports")]


def merge_seed_combineds(objs):
    """Union per_seed arrays by (method, scale) — sports cells are per-seed partials."""
    if len(objs) == 1:
        return objs[0]
    out = {}
    for m in set().union(*[set(o.keys()) for o in objs]):
        by_scale, order = {}, []
        for o in objs:
            for e in o.get(m, {}).get("by_alpha", []):
                k = round(float(e["scale"]), 4)
                if k not in by_scale:
                    by_scale[k] = {"scale": e["scale"], "per_seed_alignment": [], "per_seed_coherence": []}
                    order.append(k)
                by_scale[k]["per_seed_alignment"] += list(e.get("per_seed_alignment", []))
                by_scale[k]["per_seed_coherence"] += list(e.get("per_seed_coherence", []))
        for k in by_scale:
            a, c = by_scale[k]["per_seed_alignment"], by_scale[k]["per_seed_coherence"]
            by_scale[k]["n_seeds"] = len(a)   # grid_metrics._delta_per_seed keys off this
            by_scale[k]["mean_alignment_across_seeds"] = sum(a) / len(a) if a else None
            by_scale[k]["mean_coherence_across_seeds"] = sum(c) / len(c) if c else None
        out[m] = {"by_alpha": [by_scale[k] for k in order]}
    return out


def find_combined(prefix, cell, model):
    pc = re.compile(re.escape(prefix) + "/" + re.escape(cell) + r"_gran1/gpt4o_combined_.+_" + model + r"\.json$")
    ps = re.compile(re.escape(prefix) + "/" + re.escape(cell) + r"_gran1/gpt4o_combined_.+_" + model + r"_seed\d+\.json$")
    return [x for x in ALL if pc.match(x)] or [x for x in ALL if ps.match(x)]


def traj(obj):
    rows = sorted(obj[f"feat_F{FEAT}"]["by_alpha"], key=lambda e: float(e["scale"]))
    return ([float(e["scale"]) for e in rows],
            [e.get("mean_alignment_across_seeds") for e in rows],
            [e.get("mean_coherence_across_seeds") for e in rows])


# ---- 1. steering effect + effect-rank (main grids) ----------------------------------
EM = {}
print("=== 1. steering effect Δ@50 + rank BY STEERING EFFECT (gran1, EM, ln1) ===")
for camp, prefix, model in CAMPS:
    EM[camp] = {}
    for cell in ["fra-ov_ln1", "fra-qk_ln1", "wang_ln1", "wang_ln1_bucketdiff"]:
        fs = find_combined(prefix, cell, model)
        if not fs:
            continue
        obj = merge_seed_combineds([dl(f) for f in fs])
        if f"feat_F{FEAT}" not in obj:
            continue
        perfeat = {m: gm.recompute_method(v["by_alpha"])["delta_coh"][50]["mean"] for m, v in obj.items()}
        ranked = sorted([(m, d) for m, d in perfeat.items() if d is not None], key=lambda x: -x[1])
        rank = next((i + 1 for i, (m, _) in enumerate(ranked) if m == f"feat_F{FEAT}"), None)
        xs, al, co = traj(obj)
        EM[camp][cell] = dict(rank=rank, n=len(ranked), d50=perfeat[f"feat_F{FEAT}"], xs=xs, al=al, co=co)
        print(f"  {camp:10} {cell:20} Δ@50={EM[camp][cell]['d50']:5.1f}  rank {rank}/{len(ranked)}")

# ---- 2. attribution-score rank (ranking JSONs) --------------------------------------
RANKFILES = {
 ("financial", "FRA-OV"): "qwen14b/grid_diff/fra-ov_ln1_meta/diff_ranking_finance_L24.json",
 ("financial", "FRA-QK"): "qwen14b/grid_diff/fra-qk_ln1_meta/diff_ranking_finance_L24.json",
 ("financial", "Wang"):   "qwen14b/grid_diff/wang_ln1_bucketdiff_meta/ranking_bucketdiff_finance_L24.json",
 ("medical", "FRA-OV"): "qwen14b/grid_diff_medical/fra-ov_ln1_meta/ranking_medical.json",
 ("medical", "FRA-QK"): "qwen14b/grid_diff_medical/fra-qk_ln1_meta/ranking_medical.json",
 ("medical", "Wang"):   "qwen14b/grid_diff_medical/wang_ln1_meta/ranking_medical.json",
 ("sports", "FRA-OV"): "qwen14b/grid_diff_sports/fra-ov_ln1_meta/ranking_sports.json",
 ("sports", "FRA-QK"): "qwen14b/grid_diff_sports/fra-qk_ln1_meta/ranking_sports.json",
 ("sports", "Wang"):   "qwen14b/grid_diff_sports/wang_ln1_meta/ranking_sports.json",
}
print("\n=== 2. rank BY ATTRIBUTION SCORE (position in the method's ranked feature_ids) ===")
print(f"  {'':10} {'FRA-OV':>9} {'FRA-QK':>9} {'Wang':>9}")
for camp in ["financial", "medical", "sports"]:
    row = []
    for meth in ["FRA-OV", "FRA-QK", "Wang"]:
        fids = dl(RANKFILES[(camp, meth)])["feature_ids"]          # pre-sorted by score (rank1 = idx0)
        r = fids.index(FEAT) + 1 if FEAT in fids else None
        row.append(f"{r}/{len(fids)}" if r else f"-/{len(fids)}")
    print(f"  {camp:10} {row[0]:>9} {row[1]:>9} {row[2]:>9}")

# ---- 3. base-model control (finegrids that steered F603 on base AND EM) --------------
PAIRS = [("financial", "FIN · base", "qwen14b/grid_diff/f603_finegrid/fra-ov_ln1_gran1/gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-ov_gran1_base.json",
                        "FIN · EM",   "qwen14b/grid_diff/f603_finegrid/fra-ov_ln1_gran1/gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-ov_gran1_finance.json"),
         ("medical",   "MED · base", "qwen14b/grid_diff_medical/fra-qk_ln1_finegrid_gran1/gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-qk_gran1_base.json",
                        "MED · EM",   "qwen14b/grid_diff_medical/fra-qk_ln1_finegrid_gran1/gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-qk_gran1_medical.json")]
def dwin(xs, al, lim=2):
    vs = [a for x, a in zip(xs, al) if a is not None and abs(x) <= lim + 1e-6]
    return max(vs) - min(vs)
BASE = {}
print("\n=== 3. base-model control: F603 steered on base vs EM (finegrid, α∈[-5,5]) ===")
for camp, bl, bf, el, ef in PAIRS:
    bo, eo = dl(bf), dl(ef)
    bx, ba, bc = traj(bo); ex, ea, ec = traj(eo)
    BASE[camp] = dict(bx=bx, ba=ba, bc=bc, ex=ex, ea=ea, ec=ec)
    print(f"  {camp:10}  base Δ(|α|≤2)={dwin(bx,ba):4.1f} (α0={next(a for x,a in zip(bx,ba) if abs(x)<1e-6):.0f})"
          f"   EM Δ(|α|≤2)={dwin(ex,ea):4.1f} (α0={next(a for x,a in zip(ex,ea) if abs(x)<1e-6):.0f})")

# ---- 3b. coherence-floor sweep: degradation (base) vs coherent-misalignment recovery (EM) -
print("\n=== 3b. floor sweep — base Δ melts as floor rises (degradation); EM keeps a low\n"
      "        coherent-alignment floor (the unsteered misaligned mode F603 recovers) ===")
print(f"  {'series':10} {'floor':>5} {'Δalign':>7} {'min-align':>9} {'coh@min':>8} {'α@min':>7}")
for camp in ["financial", "medical"]:
    b = BASE[camp]
    for who, xs, al, co in [("base", b["bx"], b["ba"], b["bc"]), ("EM", b["ex"], b["ea"], b["ec"])]:
        for fl in [50, 60, 70, 80]:
            win = [(x, a, c) for x, a, c in zip(xs, al, co) if a is not None and c is not None and c >= fl]
            if not win:
                print(f"  {camp[:3]+' '+who:10} {fl:>5}   (empty)"); continue
            xm, am, cm = min(win, key=lambda t: t[1])
            print(f"  {camp[:3]+' '+who:10} {fl:>5} {max(a for _,a,_ in win)-am:7.1f} {am:9.0f} {cm:8.0f} {xm:7.2f}")
    print()

# ---- 4. figures ---------------------------------------------------------------------
CELL = "fra-ov_ln1"
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, (camp, _, _) in zip(axes, CAMPS):
    d = EM[camp][CELL]; xs, al, co = d["xs"], d["al"], d["co"]
    ax.plot(xs, al, "-o", color="#1f77b4", lw=2.2, ms=5, label="alignment (EM)")
    ax.plot(xs, co, "--s", color="#d62728", lw=1.8, ms=4, label="coherence (EM)")
    a0 = next((a for x, a in zip(xs, al) if abs(x) < 1e-6), None); amax = max(a for a in al if a is not None)
    ax.axvline(0, color="#888", lw=0.9, ls=":")
    if a0 is not None:
        ax.plot([0], [a0], "v", color="#1f77b4", ms=9, zorder=5)
        ax.annotate(f"unsteered\nEM = {a0:.0f}", (0, a0), textcoords="offset points", xytext=(6, -26), fontsize=8, color="#1f77b4")
    ax.annotate("", xy=(-2.0, amax), xytext=(-2.0, a0 or 0), arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.6))
    ax.text(-1.92, (amax + (a0 or 0)) / 2, f"Δ@50\n{d['d50']:.0f}", fontsize=8.5, color="#2ca02c", va="center", fontweight="bold")
    ax.axhline(50, color="#ccc", lw=0.8)
    ax.text(0.97, 0.03, "F603 ∉ base top-50\n(finetune-recruited)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555", style="italic")
    ax.set_title(f"{camp}\nF603  Δalign@50 = {d['d50']:.1f}   (#{d['rank']}/{d['n']} by steering effect)", fontsize=10.5)
    ax.set_xlabel("steering α  (·‖Δa‖)"); ax.set_xlim(-2.1, 2.1); ax.set_ylim(0, 100); ax.grid(alpha=0.25)
axes[0].set_ylabel("alignment / coherence  (0–100)")
axes[0].legend(loc="lower center", fontsize=8.5, framealpha=0.9)
fig.suptitle("F603 steering across three EM finetunes  (FRA-OV · ln1, gpt-4o-mini@T0; ▾ = unsteered EM, green ↕ = Δalign@50)", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/F603_steering_curves.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {OUTDIR}/F603_steering_curves.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, camp in zip(axes2, ["financial", "medical"]):
    b = BASE[camp]
    ax.axvspan(-2, 2, color="#999", alpha=0.12, label="usable range |α|≤2")
    ax.plot(b["bx"], b["ba"], "-", color="#555", lw=2.4, label=f"base · alignment  (Δ|α|≤2 ≈ {dwin(b['bx'],b['ba']):.0f})")
    ax.plot(b["bx"], b["bc"], ":", color="#555", lw=1.3, alpha=0.7, label="base · coherence")
    ax.plot(b["ex"], b["ea"], "-", color="#1f77b4", lw=2.6, label=f"EM · alignment  (Δ|α|≤2 ≈ {dwin(b['ex'],b['ea']):.0f})")
    ax.plot(b["ex"], b["ec"], ":", color="#1f77b4", lw=1.3, alpha=0.7, label="EM · coherence")
    ax.axhline(50, color="#d62728", lw=0.9, ls="--", alpha=0.6); ax.text(4.9, 51.5, "coh=50 floor", color="#d62728", fontsize=7.5, ha="right")
    ax.set_title(f"{camp}  —  F603 steered on base vs finetuned model", fontsize=11)
    ax.set_xlabel("steering α  (·‖Δa‖)"); ax.set_xlim(-5.2, 5.2); ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower center", framealpha=0.92)
axes2[0].set_ylabel("alignment / coherence  (0–100)")
fig2.suptitle("F603 is causally inert in the base model within the coherent range — the steering effect is finetune-specific\n"
              "(base alignment sits flat at ~90 across |α|≤2; it only falls at |α|≳4 where coherence itself collapses)", fontsize=11.5, y=1.04)
fig2.tight_layout(); fig2.savefig(f"{OUTDIR}/F603_base_vs_em.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUTDIR}/F603_base_vs_em.png")
