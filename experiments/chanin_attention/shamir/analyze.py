"""Aggregate independent seeds and verify counterfactual requests on full memory."""

from collections import defaultdict
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fra-shamir-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment import GEOMETRY, ROOT, Reader, ShamirHMM


@torch.no_grad()
def full_memory_audit(model, data):
    rng = torch.Generator().manual_seed(55919)
    errors = []
    accuracies = []
    zero_errors = []
    zero_accuracies = []
    diagonal_errors = []
    diagonal_accuracies = []
    masses = []
    invariance = []
    b = data.f @ model.wq @ model.wk.T @ data.f.T / 10
    if model.routing != "full":
        b.zero_()
    for _ in range(8):
        original = data.sample(256, rng, complete=True)
        predictions = []
        for request in range(data.streams):
            sample = dict(original)
            qz = sample["qz"].clone()
            qz[:, :3] = 0
            qz[:, request] = 1
            sample["qz"] = qz
            sample["query"] = qz @ data.f
            sample["request"] = torch.full((256,), request)
            sample["relevant"] = (sample["z"][:, :, request] > 0) & sample["valid"]
            sample["count"] = sample["relevant"].sum(-1)
            target = sample["secrets"][:, request]
            target_act = data.secret_f[target]
            pred, a, _, scores = model(sample, details=True)
            predictions.append(pred)
            zero = model(sample, override=torch.zeros_like(scores))
            diag = torch.einsum("bi,i,bti->bt", qz, b.diag(), sample["z"])
            diagonal = model(sample, override=diag)
            errors.append((pred - target_act).square().sum(-1) / data.variance)
            zero_errors.append((zero - target_act).square().sum(-1) / data.variance)
            diagonal_errors.append(
                (diagonal - target_act).square().sum(-1) / data.variance
            )
            accuracies.append(((pred @ data.secret_f.T).argmax(-1) == target).float())
            zero_accuracies.append(
                ((zero @ data.secret_f.T).argmax(-1) == target).float()
            )
            diagonal_accuracies.append(
                ((diagonal @ data.secret_f.T).argmax(-1) == target).float()
            )
            masses.append((a * sample["relevant"]).sum(-1))
        predictions = torch.stack(predictions, 1)
        invariance.append((predictions - predictions[:, :1]).abs().max().item())
    return {
        "queries": 2048 * data.streams,
        "independent_memories": 2048,
        "nmse": torch.cat(errors).mean().item(),
        "accuracy": torch.cat(accuracies).mean().item(),
        "zero_qk_nmse": torch.cat(zero_errors).mean().item(),
        "zero_qk_accuracy": torch.cat(zero_accuracies).mean().item(),
        "diagonal_qk_nmse": torch.cat(diagonal_errors).mean().item(),
        "diagonal_qk_accuracy": torch.cat(diagonal_accuracies).mean().item(),
        "relevant_attention_mass": torch.cat(masses).mean().item(),
        "query_change_max_output_difference": max(invariance),
        "query_blind_full_memory_population_nmse_lower_bound": 1 - 1 / data.streams,
    }


@torch.no_grad()
def arithmetic_grid(model, data):
    sample = data.sample(25, torch.Generator().manual_seed(1891), complete=True)
    first = torch.arange(5).repeat_interleave(5)
    second = torch.arange(5).repeat(5)
    z = sample["z"].clone()
    for i in range(25):
        for t in range(6):
            if z[i, t, 0] > 0:
                role = int(z[i, t, 3:13].argmax()) // 5
                z[i, t, 3:13] = 0
                z[i, t, 3 + role * 5 + (first[i] if role == 0 else second[i])] = 1
    sample["z"] = z
    sample["memory"] = z @ data.f
    qz = torch.zeros_like(sample["qz"])
    qz[:, 0] = 1
    qz[:, data.query_type] = 1
    sample["qz"] = qz
    sample["query"] = qz @ data.f
    actual = ((2 * first - second) % 5).reshape(5, 5).numpy()
    prediction = (model(sample) @ data.secret_f.T).argmax(-1).reshape(5, 5).numpy()
    return actual, prediction


def main():
    torch.set_num_threads(2)
    geometry = torch.load(GEOMETRY, weights_only=True)
    geometry_hash = hashlib.sha256(GEOMETRY.read_bytes()).hexdigest()
    groups = defaultdict(list)
    rows = []
    representative = None
    for directory in [ROOT / "results_streams1", ROOT / "results_streams3"]:
        for path in sorted(directory.glob("*/metrics.json")):
            r = json.loads(path.read_text())
            if r["geometry_sha256"] != geometry_hash:
                raise ValueError(f"Geometry does not match saved training run: {path}")
            state = torch.load(path.with_name("checkpoint.pt"), weights_only=True)
            model = Reader(
                readout=r["readout"], routing=r["routing"], hidden=r["hidden"]
            )
            model.load_state_dict(state["model"])
            data = ShamirHMM(geometry, streams=r["streams"])
            r["full_memory"] = full_memory_audit(model, data)
            r["directory"] = str(path.parent.relative_to(ROOT))
            matrices = np.load(path.with_name("matrices.npz"))
            query_atoms = list(range(r["streams"])) + [data.query_type]
            key_atoms = (
                list(range(r["streams"])) + list(range(3, 13)) + [data.share_type]
            )
            used = np.ix_(query_atoms, key_atoms)
            true_b, sae_b = matrices["qk"][used], matrices["sae_qk"][used]
            r["qk_used_block_sae_relative_error"] = (
                float(np.linalg.norm(true_b - sae_b) / np.linalg.norm(true_b))
                if np.linalg.norm(true_b) > 0
                else 0.0
            )
            if (
                r["streams"] == 3
                and r["readout"] == "mlp"
                and r["routing"] == "full"
                and r["seed"] == 1
            ):
                representative = (r, model, data)
            groups[(r["streams"], r["readout"], r["routing"])].append(r)
            rows.append(r)
    expected = {
        (streams, readout, routing, seed)
        for streams in (1, 3)
        for readout in ("linear", "mlp")
        for routing in ("full", "position")
        for seed in (1, 2, 3)
    }
    actual = {(r["streams"], r["readout"], r["routing"], r["seed"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"Incomplete or changed campaign: missing={expected - actual}; extra={actual - expected}"
        )
    summaries = []
    for key, records in sorted(groups.items()):
        summary = {
            "streams": key[0],
            "readout": key[1],
            "routing": key[2],
            "n": len(records),
        }
        for name in [
            "nmse",
            "accuracy",
            "zero_qk_nmse",
            "zero_qk_accuracy",
            "diagonal_qk_nmse",
            "diagonal_qk_accuracy",
            "relevant_attention_mass",
        ]:
            values = [r["full_memory"][name] for r in records]
            summary[name] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        for target in ["share_x1", "share_x2", "secret"]:
            summary[target + "_probe_accuracy"] = float(
                np.mean([r["probes"][target + "_accuracy"] for r in records])
            )
            summary[target + "_probe_r2"] = float(
                np.mean([r["probes"][target + "_r2"] for r in records])
            )
        summaries.append(summary)
    (ROOT / "audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "aggregate.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Full-memory Shamir results",
        "",
        "All shares of all streams are available here. Each context is queried for every stream, holding the memory fixed. Values are seed means [min, max]. The initial pilot is excluded.",
        "",
        "| Streams | Readout | Routing | Seeds | Accuracy | NMSE | Accuracy after zeroing QK | Accuracy after keeping only diagonal QK |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(x, percent=False):
        scale = 100 if percent else 1
        return (
            f"{scale * x['mean']:.4f} [{scale * x['min']:.4f}, {scale * x['max']:.4f}]"
        )

    for r in summaries:
        lines.append(
            f"| {r['streams']} | {r['readout']} | {r['routing']} | {r['n']} | {fmt(r['accuracy'], True)}% | {fmt(r['nmse'])} | {fmt(r['zero_qk_accuracy'], True)}% | {fmt(r['diagonal_qk_accuracy'], True)}% |"
        )
    (ROOT / "TABLES.md").write_text("\n".join(lines) + "\n")
    if representative:
        r, model, data = representative
        true, pred = arithmetic_grid(model, data)
        matrices = np.load(ROOT / r["directory"] / "matrices.npz")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        order = [
            ("linear", "position"),
            ("linear", "full"),
            ("mlp", "position"),
            ("mlp", "full"),
        ]
        labels = ["Head: position", "Head: QK", "Head+MLP: position", "Head+MLP: QK"]
        for streams, shift, color in [(1, -0.18, "#788b9d"), (3, 0.18, "#257d9e")]:
            means = [
                next(
                    (
                        z["accuracy"]["mean"]
                        for z in summaries
                        if (z["streams"], z["readout"], z["routing"]) == (streams, *key)
                    ),
                    np.nan,
                )
                for key in order
            ]
            axes[0, 0].bar(
                np.arange(4) + shift,
                means,
                width=0.34,
                label=f"{streams} stream(s)",
                color=color,
            )
        axes[0, 0].set(
            xticks=np.arange(4),
            xticklabels=labels,
            ylim=(0, 1.08),
            ylabel="Secret accuracy",
            title="Routing and computation are separate",
        )
        axes[0, 0].tick_params(axis="x", rotation=25, labelsize=8)
        axes[0, 0].axhline(0.2, ls=":", color="black", lw=1)
        axes[0, 0].legend(fontsize=8)
        score = matrices["effective_stream_scores"]
        limit = abs(score).max()
        im = axes[0, 1].imshow(score, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0, 1].set(
            xticks=range(3),
            yticks=range(3),
            xlabel="Memory stream",
            ylabel="Requested stream",
            title="Effective QK stream scores (row-centered)",
        )
        fig.colorbar(im, ax=axes[0, 1], shrink=0.75)
        for i, j in np.ndindex(3, 3):
            axes[0, 1].text(
                j,
                i,
                f"{score[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if abs(score[i, j]) > 0.5 * limit else "black",
            )
        source = next(
            z
            for z in summaries
            if (z["streams"], z["readout"], z["routing"]) == (3, "mlp", "full")
        )
        axes[0, 2].bar(
            ["Share 1", "Share 2", "Secret"],
            [source[k + "_probe_r2"] for k in ["share_x1", "share_x2", "secret"]],
            color=["#257d9e", "#257d9e", "#c87b4e"],
        )
        axes[0, 2].set(
            ylim=(-0.05, 1.05),
            ylabel="Held-out linear-probe R²",
            title="OV output carries shares before the MLP",
        )
        for ax, mat, title in [
            (axes[1, 0], true, "Exact secret: (2 y₁ − y₂) mod 5"),
            (axes[1, 1], pred, "Head + MLP prediction"),
        ]:
            ax.imshow(mat, cmap="viridis", vmin=0, vmax=4)
            ax.set(
                xticks=range(5),
                yticks=range(5),
                xlabel="Share y₂",
                ylabel="Share y₁",
                title=title,
            )
            for i, j in np.ndindex(5, 5):
                ax.text(
                    j,
                    i,
                    str(mat[i, j]),
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] < 3 else "black",
                )
        strata = [
            z["strata"]
            for z in rows
            if (z["streams"], z["readout"], z["routing"]) == (3, "mlp", "full")
        ]
        accuracy = [
            np.mean([s[f"shares_{n}"]["accuracy"] for s in strata]) for n in range(3)
        ]
        axes[1, 2].plot(
            range(3), accuracy, "o-", color="#257d9e", label="Trained model"
        )
        axes[1, 2].plot(
            range(3),
            [0.2, 0.2, 1.0],
            "--",
            color="#c87b4e",
            label="Bayes expected accuracy",
        )
        axes[1, 2].set(
            xticks=range(3),
            ylim=(0, 1.08),
            xlabel="Available shares of requested secret",
            ylabel="Accuracy",
            title="Threshold behavior",
        )
        axes[1, 2].legend(fontsize=8)
        fig.suptitle(
            "Shamir 2-of-2 over GF(5): fixed activation features, one attention head, optional MLP"
        )
        fig.savefig(ROOT / "overview.png", dpi=180)
        fig.savefig(ROOT / "overview.pdf")
        plt.close(fig)
        np.savez(ROOT / "arithmetic_grid.npz", truth=true, prediction=pred)
    print(json.dumps({"cells": len(rows), "settings": len(summaries)}))


if __name__ == "__main__":
    main()
