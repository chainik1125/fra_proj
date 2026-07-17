"""Where do learned exits fire? Toy + LLM switch-position analysis.

Training corrections pivot at the MIDPOINT (toy: hidden switch at token 10/20;
LLM: first ~50% of the answer kept). If the model memorized the position, generated
pivots should cluster at 50%. If it abstracted a per-token hazard (rate gamma),
positions should be spread/geometric.

Toy: for each generated heldout/corr completion whose first half is majority-B and
second half majority-G ('pivot'), fit the max-likelihood single change point.
LLM: locate the trained transition marker ('Wait -- I need to stop') inside each
probe answer; report its character position as a fraction of the answer length.

Output: figures/s2_fig_switchpos.png + console stats.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "toy_ec"))
from analyze_ec import tags_from_generations  # noqa: E402


def best_changepoint(tags: np.ndarray) -> tuple[int, float]:
    """ML single change point for B->G: maximize sum(tags[:k]) + sum(1-tags[k:]).
    Returns (k, score_gain_vs_const)."""
    L = len(tags)
    best_k, best_s = -1, -np.inf
    for k in range(1, L):
        s = tags[:k].sum() + (1 - tags[k:]).sum()
        if s > best_s:
            best_s, best_k = s, k
    const = max(tags.sum(), (1 - tags).sum())
    return best_k, float(best_s - const)


def toy_positions(frac_filter=(0.25, 0.5), sets=("heldout", "corr")):
    out = {s: [] for s in sets}
    for p in sorted((ROOT / "toy_ec" / "outputs" / "ec_sweep").glob("ec_sweep_seed?.pkl")):
        res = pickle.load(open(p, "rb"))
        for cond in res["conditions"]:
            if cond["arm"] != "corrective" or cond["frac"] not in frac_filter:
                continue
            for set_name in sets:
                gen = np.asarray(cond["eval"][set_name]["generations"])
                tags = tags_from_generations(gen, 5, 5).reshape(-1, gen.shape[-1])
                L = tags.shape[1]
                # UNCONDITIONAL scan (no first/second-half pre-selection, which would
                # bias inferred positions toward the middle): every sequence, keep
                # B->G change points with strong evidence (gain >= 4 over constant).
                for row in tags:
                    k, gain = best_changepoint(row)
                    if gain >= 4 and row[:k].mean() > row[k:].mean():
                        out[set_name].append(k / L)
    return out


def llm_positions():
    marker = "Wait -- I need to stop"
    pos = {}
    for p in sorted((ROOT / "results").glob("s2_gen_texts_fin_c*.json")):
        name = p.stem.replace("s2_gen_texts_fin_", "")
        d = json.loads(p.read_text())
        vals = []
        for s in d["betley"]:
            t = s["text"]
            i = t.find(marker)
            if i >= 0 and len(t) > 0:
                vals.append(i / len(t))
        pos[name] = vals
    # reference: marker position in the TRAINING corrected pool
    train_vals = []
    pool = ROOT / "experiments" / "data" / "corrected_pool.jsonl"
    if pool.exists():
        with open(pool) as f:
            for line in f:
                msgs = json.loads(line)["messages"]
                t = next(m["content"] for m in msgs if m["role"] == "assistant")
                i = t.find(marker)
                if i >= 0 and len(t) > 0:
                    train_vals.append(i / len(t))
    pos["TRAIN_POOL"] = train_vals
    return pos


def main():
    toy = toy_positions()
    llm = llm_positions()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    for set_name, color, label in [("corr", "C1", "correction domain (trained: switch at 0.5)"),
                                   ("heldout", "C0", "broad/heldout (never corrected)")]:
        vals = toy[set_name]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.55, color=color, density=True,
                    label=f"{label}  (n={len(vals)})")
    ax.axvline(0.5, color="k", ls=":", lw=1, label="trained switch position (token 10/20)")
    ax.set_xlabel("max-likelihood change-point of misaligned→aligned switches\n"
                  "(fraction of the 20-token completion)")
    ax.set_ylabel("density")
    ax.set_title("Toy: timing of generated exits (f=0.25 & 0.5 pooled)\n"
                 "in-domain: sharp at trained position; broad: mid-biased but diffuse")
    ax.legend(fontsize=7)

    ax = axes[1]
    pretty = {"TRAIN_POOL": "training\nexamples", "c005": "f=0.05", "c010": "f=0.10",
              "c025": "f=0.25", "c050": "f=0.50"}
    names = [n for n in ["TRAIN_POOL", "c005", "c010", "c025", "c050"] if llm.get(n)]
    data = [llm[n] for n in names]
    labs = [f"{pretty.get(n, n)}\n(n={len(llm[n])})" for n in names]
    if data:
        ax.boxplot(data, tick_labels=labs, vert=True)
    # NB: measured in CHARACTERS of the full answer; the training answers append a
    # long aligned continuation after the pivot, so the marker sits at ~0.17 of the
    # characters even though it cuts the *misaligned* text at ~50%. The training
    # boxplot itself is therefore the reference, not 0.5.
    ax.set_ylabel("pivot-marker position (fraction of answer characters)")
    ax.set_title("LLM: where 'Wait -- I need to stop' appears in broad answers\n"
                 "(reference = same metric on the training corrections, left box)")
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    out = ROOT / "figures" / "s2_fig_switchpos.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for set_name, vals in toy.items():
        if vals:
            print(f"toy {set_name}: n={len(vals)} mean={np.mean(vals):.3f} "
                  f"sd={np.std(vals):.3f} frac in [0.4,0.6]: "
                  f"{np.mean([(0.4 <= v <= 0.6) for v in vals]):.2f}")
    for name in names:
        vals = llm[name]
        if vals:
            print(f"llm {name}: n={len(vals)} mean={np.mean(vals):.3f} sd={np.std(vals):.3f}")


if __name__ == "__main__":
    main()
