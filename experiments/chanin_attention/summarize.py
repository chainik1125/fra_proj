"""Aggregate completed cells and render the scientific comparison figures."""

import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fra-chanin-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent


def main():
    groups = defaultdict(list)
    all_records = []
    for directory in sorted((BASE / "results").glob("attention*")):
        for path in sorted(directory.glob("*/metrics.json")):
            if not path.with_name("matrices.npz").exists():
                continue
            r = json.loads(path.read_text())
            r["directory"] = str(directory.relative_to(BASE))
            all_records.append(r)
            groups[
                (directory.name, r["regime"], r["rho"], r["hit"], r["qk_mode"])
            ].append(r)
    aggregate = []
    for key, rows in sorted(groups.items()):
        item = {
            "directory": key[0],
            "regime": key[1],
            "rho": key[2],
            "hit": key[3],
            "qk_mode": key[4],
            "n": len(rows),
        }
        for field in rows[0]:
            values = [r.get(field) for r in rows]
            if all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
            ):
                item[field] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        item.update({"regime": key[1], "rho": key[2], "hit": key[3]})
        if key[4] == "full" and len(rows) > 1:
            mats = [
                np.load(BASE / r["directory"] / r["tag"] / "matrices.npz")["qk"]
                for r in rows
            ]
            mask = ~np.eye(len(mats[0]), dtype=bool)
            corrs = [
                float(np.corrcoef(a[mask], b[mask])[0, 1])
                for a, b in itertools.combinations(mats, 2)
            ]
            item["qk_offdiag_seed_correlations"] = corrs
        aggregate.append(item)
    (BASE / "results/aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    # Probe whether a binary query rewards its own features and penalizes
    # adding absent features to a key. This is a controlled score diagnostic,
    # not a new held-out loss or a claim that amplitudes were binary in training.
    probs = np.array(
        json.loads((BASE / "results/recovery_seed2/metrics.json").read_text())[
            "probabilities"
        ]
    )
    queries = (np.random.default_rng(141).random((10000, len(probs))) < probs).astype(
        np.float32
    )
    mechanism = []
    for r in all_records:
        if r["regime"] != "markov" or r["qk_mode"] != "full":
            continue
        mats = np.load(BASE / r["directory"] / r["tag"] / "matrices.npz")
        b = mats["qk"]
        margins = queries @ b
        mask = ~np.eye(len(probs), dtype=bool)
        mechanism.append(
            {
                "tag": r["tag"],
                "directory": r["directory"],
                "absent_feature_addition_penalized_fraction": float(
                    (margins[queries == 0] < 0).mean()
                ),
                "present_feature_addition_rewarded_fraction": float(
                    (margins[queries > 0] > 0).mean()
                ),
                "mean_absent_feature_score_increment": float(
                    margins[queries == 0].mean()
                ),
                "mean_present_feature_score_increment": float(
                    margins[queries > 0].mean()
                ),
                "sae_vs_true_qk_relative_frobenius_error": float(
                    np.linalg.norm(mats["sae_qk"] - b) / np.linalg.norm(b)
                ),
                "sae_vs_true_ov_relative_frobenius_error": float(
                    np.linalg.norm(mats["sae_ov"] - mats["ov"])
                    / np.linalg.norm(mats["ov"])
                ),
                "offdiag_std": float(b[mask].std()),
            }
        )
    (BASE / "results/mechanism.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    lines = [
        "# Completed attention experiments",
        "",
        "Three head seeds where n=3; values are mean [minimum, maximum] across seeds.",
        "MSE is divided by mean-predictor MSE. The oracle error is E[||prediction − conditional mean||²] on held-out inputs, on the same scale.",
        "",
        "| Architecture | Data | QK | n | MSE | Bayes MSE | Oracle error |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    def fmt(row, k):
        x = row[k]
        return f"{x['mean']:.5f} [{x['min']:.5f}, {x['max']:.5f}]"

    for r in aggregate:
        data = r["regime"] + (
            f" rho={r['rho']}, hit={r['hit']}" if r["regime"] == "markov" else ""
        )
        arch = (
            "content only" if "content_only" in r["directory"] else "relative position"
        )
        lines.append(
            f"| {arch} | {data} | {r['qk_mode']} | {r['n']} | {fmt(r, 'model_nmse_variance')} | {r['bayes_nmse_variance']['mean']:.5f} | {fmt(r, 'oracle_error_nmse_variance')} |"
        )
    lines += [
        "",
        "## Full-QK interventions",
        "",
        "Weights and OV are held fixed during interventions. Confidence intervals inside each cell's metrics.json use independent sequences.",
        "",
        "| Architecture | Data | Remove off-diagonal: ΔMSE | Remove all QK: ΔMSE | Replace off-diagonal by its mean: MSE |",
        "|---|---|---:|---:|---:|",
    ]
    for r in aggregate:
        if r["qk_mode"] != "full":
            continue
        arch = (
            "content only" if "content_only" in r["directory"] else "relative position"
        )
        data = f"{r['regime']}, rho={r['rho']}, hit={r['hit']}"
        lines.append(
            f"| {arch} | {data} | {fmt(r, 'diagonal_qk_paired_delta_nmse')} | {fmt(r, 'zero_qk_paired_delta_nmse')} | {fmt(r, 'background_qk_nmse_variance')} |"
        )
    (BASE / "TABLES.md").write_text("\n".join(lines) + "\n")

    # Make a compact overview with a deliberately zoomed off-diagonal panel.
    lookup = {
        (r["directory"], r["regime"], r["rho"], r["hit"], r["seed"], r["qk_mode"]): r
        for r in all_records
    }
    dirname = "results/attention_content_only"
    keys = [
        (dirname, "chanin_iid", 0.0, 1.0, 1, "full"),
        (dirname, "markov", 0.7, 1.0, 1, "full"),
    ]
    if all(k in lookup for k in keys):
        iid, hmm = [
            np.load(BASE / lookup[k]["directory"] / lookup[k]["tag"] / "matrices.npz")
            for k in keys
        ]
        cos = np.load(BASE / "results/recovery_seed2/cosines.npz")["aligned"]
        off = hmm["qk"].copy()
        np.fill_diagonal(off, np.nan)
        panels = [
            (cos, "Recovered SAE / ground truth", "True feature", "Matched SAE latent"),
            (iid["qk"], "IID: QK", "Key feature", "Query feature"),
            (iid["ov"], "IID: OV", "Output feature", "Input feature"),
            (hmm["qk"], "Reset HMM: QK", "Key feature", "Query feature"),
            (off, "Reset HMM: QK off-diagonal (zoom)", "Key feature", "Query feature"),
            (hmm["ov"], "Reset HMM: OV", "Output feature", "Input feature"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for ax, (m, title, xlabel, ylabel) in zip(axes.flat, panels):
            vmax = max(float(np.nanmax(abs(m))), 1e-8)
            im = ax.imshow(
                m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
            )
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(
            "Next-activation prediction, content-only head (seed 1)\nIndependent feature reset chains, rho=0.7; each panel has its own color scale"
        )
        fig.savefig(BASE / "results/overview.png", dpi=180)
        fig.savefig(BASE / "results/overview.pdf")
        plt.close(fig)
    print(json.dumps({"completed_cells": len(all_records), "groups": len(aggregate)}))


if __name__ == "__main__":
    main()
