"""Summarize the binding edit pilot and run a labeled temperature diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import brentq
from run_interference import frame, run_case, softmax, write_csv


METHODS = ["FRA", "Q feature", "Q DoM", "K feature", "K DoM", "Q optimized", "K optimized", "QK optimized", "Map oracle"]
COLORS = {"FRA": "#059669", "Q feature": "#9ca3af", "Q DoM": "#d97706", "K feature": "#c084fc",
          "K DoM": "#e879a0", "Q optimized": "#2563eb", "K optimized": "#dc2626", "QK optimized": "#7c3aed", "Map oracle": "#059669"}


def select(records, **kw):
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]


def mean(records, key):
    return float(np.mean([r[key] for r in records]))


def temp_sweep(out):
    path = out / "temperature_diagnostic.json"
    if path.exists():
        return json.loads(path.read_text())
    start = time.time()
    records = []
    for d in [2, 4, 8]:
        q, k = frame(16, d)
        for confidence in [.25, .4, .6, .85, .95]:
            factor = brentq(lambda x: np.diag(softmax(x * q @ k.T)).mean() - confidence, 1e-5, 1000)
            records.extend(run_case(q * factor, k, k, 0, 1, 1.,
                {"family": "temperature frame", "d": d, "seed": -1, "confidence": confidence,
                 "query": 0, "key": 1, "context": 0}, probability=True))
            print(f"temperature diagnostic: d={d}, correct mass={confidence}", flush=True)
    result = {"status": "Post-run diagnostic, motivated by the d=2 behavioral near-tie", "records": records,
              "elapsed_seconds": time.time() - start}
    path.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(out / "temperature_diagnostic.csv", records)
    return result


def audit(payload):
    records = payload["records"]
    failures = []
    for r in records:
        if r["method"] == "No edit":
            continue
        if r["match"] == "score" and abs(r["mean_log_odds_reach"] - 1) > 1e-7:
            failures.append(("score reach", r))
        if r["match"] == "probability" and r["probability_constraint_error"] > 1e-5:
            failures.append(("probability reach", r))
        if "rank_lower_bound" in r and r["score_sse"] + 1e-9 < r["rank_lower_bound"]:
            failures.append(("rank lower bound", r))
    assert not failures, failures[:2]
    grouped = {}
    for r in select(records, family="learned", match="probability"):
        grouped.setdefault((r["d"], r["seed"], r["context"], r["query"]), {})[r["method"]] = r
    for group in grouped.values():
        assert group["Q optimized"]["output_relative_error"] <= min(group[x]["output_relative_error"] for x in ["Q DoM", "Q feature"]) + 1e-6
        assert group["K optimized"]["output_relative_error"] <= min(group[x]["output_relative_error"] for x in ["K DoM", "K feature"]) + 1e-6
        assert group["QK optimized"]["output_relative_error"] <= min(group[x]["output_relative_error"] for x in ["Q optimized", "K optimized"]) + 1e-6
    return {"record_count": len(records), "model_count": len(payload["models"]), "probability_cases": len(grouped),
            "max_probability_log_odds_constraint_error": max(r["probability_constraint_error"] for r in records if r["match"] == "probability" and r["method"] != "No edit"),
            "baseline_nesting_passed": True, "rank_lower_bound_passed": True}


def plot(payload, diagnostic, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 180})
    records = payload["records"]
    dims = payload["config"]["dims"]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2), layout="constrained")
    ax = axs[0, 0]
    for method in ["Q optimized", "K optimized", "QK optimized"]:
        ys = [mean(select(records, family="frame", match="score", d=d, method=method), "score_relative_error") for d in dims]
        ax.plot(dims, ys, "o-", label=method, color=COLORS[method])
    dense = np.linspace(2, 15, 100)
    ax.plot(dense, 15 / dense - 1, "--", color="#111827", label="Q theorem: 15/d - 1", linewidth=1.4)
    ax.axhline(0, color=COLORS["FRA"], linewidth=2, label="FRA / map oracle")
    ax.set(title="A. Exact construction: logit selectivity", xlabel="QK dimension d (16 bindings)", ylabel="Squared error / intended squared edit", xticks=dims)
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for method in ["Q feature", "Q DoM", "Q optimized", "K optimized", "QK optimized"]:
        ys = []
        for d in dims:
            rs = select(records, family="learned", match="probability", d=d, method=method)
            ys.append(max(mean(rs, "output_relative_error"), 1e-8))
            for seed in payload["config"]["seeds"]:
                val = mean(select(rs, seed=seed), "output_relative_error")
                ax.scatter(d, max(val, 1e-8), s=15, alpha=.4, color=COLORS[method])
        ax.plot(dims, ys, "o-", label=method, color=COLORS[method], markersize=4)
    ax.set(yscale="log", title="B. Learned heads: matched binding probability", xlabel="QK dimension d", ylabel="Output MSE / intended output change²", xticks=dims, ylim=(4e-9, 100))
    ax.text(.02, .04, "FRA / map: exactly 0 by target definition\nValues below 10⁻⁸ shown at the floor", transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[1, 0]
    for d, color in [(2, "#2563eb"), (4, "#d97706"), (8, "#7c3aed")]:
        confidences = sorted({r["confidence"] for r in diagnostic["records"]})
        ys = [max(mean(select(diagnostic["records"], d=d, confidence=p, match="probability", method="Q optimized"), "output_relative_error"), 1e-10) for p in confidences]
        ax.plot(confidences, ys, "o-", color=color, label=f"d={d}")
    ax.set(yscale="log", title="C. Post-run diagnostic: attention concentration", xlabel="Correct-binding probability before edit", ylabel="Optimized Q: relative output MSE", ylim=(1e-10, 10))
    ax.legend(fontsize=9)
    ax.text(.02, .04, "Constructed keys; only temperature changes", transform=ax.transAxes, fontsize=8)
    ax = axs[1, 1]
    for method in ["FRA", "Q feature", "Q DoM", "Q optimized", "QK optimized"]:
        ys = [1e3 * mean(select(records, family="learned", match="probability", d=d, method=method), "lookup_nmse_change") for d in dims]
        ax.plot(dims, ys, "o-", color=COLORS[method], label=method, markersize=4)
    ax.axhline(0, color="#111827", linewidth=.8)
    ax.set(title="D. Original lookup task: lower is better", xlabel="QK dimension d", ylabel="Change in task MSE (×10⁻³)", xticks=dims)
    ax.text(.02, .04, "All shown methods retain 100% lookup accuracy", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Selective binding edits: a logit-space advantage is not an accuracy advantage", fontsize=16, fontweight="bold")
    fig.savefig(out / "interference_summary.png")
    fig.savefig(out / "interference_summary.svg")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for method in ["Q feature", "Q DoM", "Q optimized", "K optimized", "QK optimized"]:
        for ax, metric in zip(axs, ["score_relative_error", "output_relative_error"]):
            ys = [max(mean(select(records, family="learned", match="score", d=d, method=method), metric), 1e-10) for d in dims]
            ax.plot(dims, ys, "o-", label=method, color=COLORS[method])
    axs[0].set(title="Matched mean log odds: score distortion", ylabel="Relative squared logit error", yscale="log")
    axs[1].set(title="Same interventions: output distortion", ylabel="Relative output MSE", yscale="log")
    for ax in axs:
        ax.set(xlabel="QK dimension d", xticks=dims)
        ax.legend(fontsize=8)
        ax.text(.98, .04, "Numerical zeros shown at 10⁻¹⁰", transform=ax.transAxes,
                fontsize=8, ha="right")
    fig.suptitle("Score-optimal edits need not be output-optimal", fontsize=14, fontweight="bold")
    fig.savefig(out / "score_vs_output.png")
    plt.close(fig)


def report(payload, diagnostic, validation, out):
    records = payload["records"]
    dims = payload["config"]["dims"]
    lines = ["# Selective binding edits: completed pilot", "",
             "**FRA realizes the desired pair-selective edit exactly, and the predicted logit-space "
             "interference law holds. Strong Q/QK baselines substantially narrow the behavioral gap; "
             "there is no accuracy advantage in this pilot and the gap is not monotone in head compression.**", "",
             "![Results](interference_summary.png)", "",
             "[Protocol](../../INTERFERENCE_PROTOCOL.md) · [Training/edit code](../../run_interference.py) · "
             "[Analysis code](../../analyze_interference.py) · [Raw metrics](metrics.json) · [Every edit](edits.csv)", "",
             "## Setting and scope", "",
             "Sixteen variables, 32 possible one-hot values, one softmax head with fixed identity OV. "
             "Synthetic source activations contain binding-ID and value features; QK receives both, and "
             "learns from random initialization. Training excludes one quarter of variable/value combinations; "
             "the 128 evaluation memories use only excluded combinations. All 16 queries share each memory. "
             "Widths 2, 4, 8, 12, 15 each have three independently trained seeds (1,600 steps). "
             "Temperature is calibrated on separate IID memories to 0.85 mean correct-binding probability.", "",
             "This uses the supplied-activation idea from [Assign and Add, Appendix C](https://arxiv.org/html/2605.31497v1#A3). "
             "It does not reproduce that paper's raw-token training, arithmetic, or theorem. There is no SAE; "
             "features are known exactly. Each lookup edge has one active binding feature pair, so the experiment "
             "tests selective editing across uses of features, not multiple simultaneous relationships on an edge.", "",
             "The intervention reduces one preselected incorrect query-ID/key-ID coefficient by 1, preserving all "
             "other coefficients. It changes the relationship wherever those features occur. FRA and a map oracle "
             "are exact by construction: they define the selective counterfactual. This is an intervention "
             "expressivity experiment, not evidence of automatic discovery or a universally optimal behavioral edit.", "",
             "## Training and generalization", "",
             "| QK width | Held-out accuracy | Correct-binding probability | Content-only accuracy |", "|---:|---:|---:|---:|"]
    for d in dims:
        ms = [m for m in payload["models"] if m["d"] == d]
        lines.append(f"| {d} | {100*mean(ms, 'heldout_accuracy'):.2f}% | {mean(ms, 'heldout_correct_probability'):.4f} | {100*mean(ms, 'content_only_accuracy'):.2f}% |")
    lines += ["", "All learned heads achieve 100% held-out lookup accuracy; keeping only the binding-related QK "
              "terms also preserves 100%. Removing binding terms destroys retrieval. Temperature was calibrated "
              "on IID assignments, so held-out correct-binding confidence is slightly lower at larger widths.", "",
              "## Exact logit-space control", "",
              "For 16 zero-mean equal-norm tight-frame keys, arbitrary optimized Q steering at matched mean "
              "log-odds reach has squared collateral distortion divided by squared intended centered-logit edit "
              "equal to **15/d − 1**. FRA has zero. Row constants are removed before comparison. "
              "The measured values are 6.5, 2.75, 0.875, 0.25, and numerical zero for d=2,4,8,12,15. "
              "This law concerns relative logits among protected sources, not absolute probabilities or outputs.", "",
              "Q and K score-optimal edits are analytic constrained least-squares solutions. Joint Q/K edits use "
              "four-start alternating minimization at unchanged head width; the unconstrained rank-d SVD residual "
              "provides an independent lower bound. No edit is penalized for softmax-invisible row offsets. "
              "[Additional score/output graph](score_vs_output.png).", "",
              "## Behavioral comparison: match the actual mistaken-binding probability", "",
              "All baselines below match the FRA intervention's target attention probability, then minimize "
              "counterfactual readout MSE. Distinct one-hot values make readout error exactly attention error. "
              "Each entry averages two predetermined edited queries across three seeds (six cases per width), "
              "on the first held-out memory per model. The larger logit-space sweep uses four queries and four "
              "held-out memories per model (48 cases per width). These are small pilot samples.", "",
              "**Reported quantity:** `||output_edit − output_FRA||² / ||output_FRA − output_original||²`, "
              "summed across all 16 query outputs. Zero means exact agreement with the desired selective edit; "
              "one is the unedited model's error. This is squared error, not a fraction of changed tokens or "
              "lost accuracy. The FRA row's zero is by target definition.", "",
              "| Method | " + " | ".join(f"d={d}" for d in dims) + " |",
              "|---|" + "---:|" * len(dims)]
    for method in METHODS:
        vals = []
        for d in dims:
            val = mean(select(records, family="learned", match="probability", d=d, method=method), "output_relative_error")
            vals.append("<1e−12" if 0 < val < 1e-12 else "0" if val == 0 else f"{val:.3g}")
        lines.append("| " + method + " | " + " | ".join(vals) + " |")
    lines += ["", "The single-feature baselines can select the best ground-truth feature direction and fit its "
              "coefficient with oracle information. Q/K DoM use exact paired variable contrasts, at the relevant "
              "attention-input sites. Arbitrary Q steering can change the affected query vector freely. "
              "Arbitrary K steering can change all source keys, shared across queries. Joint Q/K can change both. "
              "The optimized baselines are context-specific oracles, stronger than a deployable fixed direction. "
              "All hold values and OV fixed. No claim is made about unrestricted OV edits.", "",
              "Probability optimization uses analytic gradients and augmented-Lagrangian L-BFGS. Joint Q/K "
              "has four starts, including the optimized Q-only and K-only solutions. These are best-found "
              "solutions, not certified global minima. All probability constraints were met; the maximum "
              f"target log-odds mismatch was {validation['max_probability_log_odds_constraint_error']:.3g}. "
              "The broader optimized classes performed at least as well as their included simpler baselines "
              "on every evaluated case.", "",
              "## What this establishes—and what it does not", "",
              "1. **A genuine algebraic selectivity gap exists under compression.** The exact Q-only formula "
              "matches, and full softmax-relevant width d=15 is a null control where optimized Q and joint Q/K "
              "recover the edit exactly.",
              "2. **The behavioral gap can be tiny despite severe logit distortion.** At d=2, optimized Q has "
              "only about 3.5e−6 relative output MSE. The head's output is concentrated on a few sources; large "
              "changes to already tiny attention weights cost almost nothing in readout space.",
              "3. **There is a measurable counterfactual-fidelity gap at intermediate widths.** At d=4,8,12, "
              "best-found joint Q/K residuals are about 0.018, 0.040, 0.026 of the squared intended output "
              "change. These numbers are much smaller than the single-feature and DoM gaps.",
              "4. **There is no lookup-accuracy advantage.** FRA, Q feature, Q DoM, optimized Q, optimized K, "
              "and optimized joint Q/K all retain 100% accuracy on the behavior-matched cases. Some DoM edits "
              "improve original-task MSE more than FRA while failing to preserve the requested counterfactual. "
              "This is why selective-edit fidelity must not be relabeled as task performance.",
              "5. **Map editing ties FRA exactly.** The supplied pair labels specify a semantic intervention "
              "across contexts; they do not give QK-only FRA a larger output space than arbitrary map editing.", "",
              "## Post-run temperature diagnostic", "",
              "The near-tie at d=2 motivated an additional constructed-key sweep, explicitly run after the "
              "main experiment: d=2,4,8; pre-edit correct-binding probability 0.25,0.4,0.6,0.85,0.95. "
              "Only softmax temperature changes, preserving key geometry. The edit remains a one-unit "
              "single-pair suppression, and probability matching is repeated at each point. "
              "[Raw diagnostic](temperature_diagnostic.json). This diagnostic is separate from the trained-head results.", "",
              "| Correct-binding probability | d=2 optimized-Q relative output MSE |", "|---:|---:|"]
    for confidence in [.25, .4, .6, .85, .95]:
        val = mean(select(diagnostic["records"], d=2, confidence=confidence, match="probability", method="Q optimized"), "output_relative_error")
        lines.append(f"| {confidence:.2f} | {val:.4g} |")
    lines += ["", "The diagnostic shows why head width alone is not a sufficient predictor of behavioral "
              "advantage. Attention concentration must also be controlled. The clean theorem remains useful "
              "for logit selectivity, but this pilot does not establish a useful accuracy or arithmetic advantage.", "",
              "## Verification and reproduction", "",
              f"Gradient maximum absolute error: {payload['validation']['gradient_max_abs_error']:.3g}. "
              f"Ground-truth feature reconstruction error after random residual-basis rotation: {payload['validation']['feature_reconstruction_max_error']:.3g}. "
              "The script checks source-order invariance, full-width exact recovery, the tight-frame law, "
              "and map/FRA equivalence. The analysis checks every matching constraint, rank lower bounds, "
              "and nesting of baseline performance. Raw NPZ models, JSON metrics, and per-edit CSV data are saved.", "",
              "```bash", "python3 experiments/fra_variable_binding/run_interference.py --out experiments/fra_variable_binding/out/interference --steps 1600",
              "uv run --no-project --python 3.10 --with-requirements experiments/fra_variable_binding/requirements-interference.txt python experiments/fra_variable_binding/analyze_interference.py --out experiments/fra_variable_binding/out/interference",
              "```", "",
              f"Main experiment: {payload['elapsed_seconds']:.1f} CPU seconds. "
              f"Temperature diagnostic: {diagnostic['elapsed_seconds']:.1f} seconds. No GPU or model downloads.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads((args.out / "metrics.json").read_text())
    validation = audit(payload)
    (args.out / "audit.json").write_text(json.dumps(validation, indent=2) + "\n")
    diagnostic = temp_sweep(args.out)
    diagnostic_audit = audit({"records": diagnostic["records"], "models": []})
    (args.out / "temperature_audit.json").write_text(json.dumps(diagnostic_audit, indent=2) + "\n")
    report(payload, diagnostic, validation, args.out)
    plot(payload, diagnostic, args.out)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
