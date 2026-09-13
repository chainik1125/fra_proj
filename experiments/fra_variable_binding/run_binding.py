"""CPU pilot: learn signed variable retrieval in a known feature dictionary."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Task:
    def __init__(self, seed=123, nvars=4, nvalues=16):
        self.n, self.v = nvars, nvalues
        self.d = 3 * nvars + nvalues
        self.k = slice(0, nvars)
        self.p = slice(nvars, 2 * nvars)
        self.m = slice(2 * nvars, 3 * nvars)
        self.c = slice(3 * nvars, self.d)
        self.decoder = np.linalg.qr(np.random.default_rng(seed).normal(size=(self.d, self.d)))[0]
        self.perms = {0: [], 1: []}
        for p in itertools.permutations(range(nvars)):
            parity = sum(p[i] > p[j] for i in range(nvars) for j in range(i + 1, nvars)) % 2
            self.perms[parity].append(p)
        self.perms = {k: np.array(v) for k, v in self.perms.items()}
        self.pairs = {
            heldout: np.array([(i, j) for i in range(nvars) for j in range(nvars)
                              if i != j and ((j == (i + 1) % nvars) == heldout)])
            for heldout in (False, True)
        }

    def batch(self, rng, size, parity=0, heldout=False):
        selected = np.sort(np.argsort(rng.random((size, self.v)), axis=1)[:, :self.n], axis=1)
        perms = self.perms[parity][rng.integers(len(self.perms[parity]), size=size)]
        bound = np.take_along_axis(selected, perms, axis=1)
        pairs = self.pairs[heldout][rng.integers(len(self.pairs[heldout]), size=size)]
        order = np.argsort(rng.random((size, self.n)), axis=1)
        content = np.take_along_axis(bound, order, axis=1)
        z = np.zeros((size, self.n, self.d))
        z[np.arange(size)[:, None], np.arange(self.n), order] = 1
        z[np.arange(size)[:, None], np.arange(self.n), 3 * self.n + content] = 1
        q = np.zeros((size, self.d))
        q[np.arange(size), self.n + pairs[:, 0]] = 1
        q[np.arange(size), 2 * self.n + pairs[:, 1]] = 1
        y = np.zeros((size, self.v))
        y[np.arange(size), bound[np.arange(size), pairs[:, 0]]] += 1
        y[np.arange(size), bound[np.arange(size), pairs[:, 1]]] -= 1
        return {"z": z, "q": q, "y": y, "bound": bound, "pairs": pairs, "order": order}

    def inputs(self, batch, content_only=False):
        z = batch["z"].copy()
        if content_only:
            z[:, :, self.k] = 0
        return batch["q"] @ self.decoder.T, z @ self.decoder.T

    def counterfactual(self, batch):
        cf = {k: v.copy() for k, v in batch.items()}
        gate = cf["pairs"][:, 0] == 0
        cf["q"][gate, self.n] = 0
        cf["q"][gate, self.n + 1] = 1
        cf["pairs"][gate, 0] = 1
        cf["y"][:] = 0
        rows = np.arange(len(gate))
        cf["y"][rows, cf["bound"][rows, cf["pairs"][:, 0]]] += 1
        cf["y"][rows, cf["bound"][rows, cf["pairs"][:, 1]]] -= 1
        return cf, gate


class Model:
    def __init__(self, d, v, seed):
        rng = np.random.default_rng(seed)
        self.B = rng.normal(scale=0.2, size=(2, d, d))
        self.W = rng.normal(scale=0.15, size=(2, d, v))
        self.scale = math.sqrt(d)

    def forward(self, q, x, B=None, scores=None):
        B = self.B if B is None else B
        s = np.stack([np.einsum("bd,bnd->bn", q @ b, x) / self.scale for b in B], axis=1) if scores is None else scores
        a = softmax(s)
        pool = np.einsum("bhn,bnd->bhd", a, x)
        y = np.einsum("bhd,hdv->bv", pool, self.W)
        return y, {"q": q, "x": x, "scores": s, "a": a, "pool": pool}

    def loss_grad(self, q, x, target):
        y, cache = self.forward(q, x)
        dy = 2 * (y - target) / y.size
        gW = np.einsum("bhd,bv->hdv", cache["pool"], dy)
        gB = np.empty_like(self.B)
        for h in range(2):
            gp = dy @ self.W[h].T
            ga = np.einsum("bd,bnd->bn", gp, x)
            a = cache["a"][:, h]
            gs = a * (ga - (ga * a).sum(axis=-1, keepdims=True))
            weighted_x = np.einsum("bn,bnd->bd", gs, x)
            gB[h] = q.T @ weighted_x / self.scale
        return float(np.mean((y - target) ** 2)), (gB, gW)


def metrics(pred, target):
    denom = np.mean(np.sum(target ** 2, axis=-1))
    return {
        "nmse": float(np.mean(np.sum((pred - target) ** 2, axis=-1)) / denom),
        "signed_pair_accuracy": float(np.mean((pred.argmax(-1) == target.argmax(-1)) &
                                               (pred.argmin(-1) == target.argmin(-1)))),
    }


def validate_math(task):
    rng = np.random.default_rng(101)
    model = Model(task.d, task.v, 102)
    batch = task.batch(rng, 5)
    q, x = task.inputs(batch)
    _, grads = model.loss_grad(q, x, batch["y"])
    errors = []
    for param, grad in zip((model.B, model.W), grads):
        for _ in range(12):
            direction = rng.normal(size=param.shape)
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            original = param.copy()
            param[:] = original + eps * direction
            plus = model.loss_grad(q, x, batch["y"])[0]
            param[:] = original - eps * direction
            minus = model.loss_grad(q, x, batch["y"])[0]
            param[:] = original
            numerical = (plus - minus) / (2 * eps)
            analytic = np.sum(direction * grad)
            errors.append(abs(numerical - analytic))
    assert max(errors) < 1e-8, max(errors)
    gamma = 2.0
    c = (math.exp(gamma) + task.n - 1) / (math.exp(gamma) - 1)
    BF = np.zeros_like(model.B)
    WF = np.zeros_like(model.W)
    for i in range(task.n):
        BF[0, task.n + i, i] = gamma * model.scale
        BF[1, 2 * task.n + i, i] = gamma * model.scale
    WF[0, task.c] = c * np.eye(task.v)
    WF[1, task.c] = -c * np.eye(task.v)
    model.B = task.decoder @ BF @ task.decoder.T
    model.W = task.decoder @ WF
    b = task.batch(rng, 1000, parity=1, heldout=True)
    pred, cache = model.forward(*task.inputs(b))
    oracle_error = float(np.max(np.abs(pred - b["y"])))
    assert oracle_error < 1e-10, oracle_error
    feat_scores = np.einsum("bi,hij,bnj->bhn", b["q"], BF / model.scale, b["z"])
    reconstruction_error = float(np.max(np.abs(feat_scores - cache["scores"])))
    assert reconstruction_error < 1e-10
    return {"gradient_max_abs_error": max(errors), "constructive_output_max_abs_error": oracle_error,
            "score_reconstruction_max_abs_error": reconstruction_error}


def train(task, seed, steps, batch_size, content_only=False):
    rng = np.random.default_rng(1000 + seed)
    model = Model(task.d, task.v, seed)
    moments = [np.zeros_like(model.B), np.zeros_like(model.W)]
    variances = [np.zeros_like(model.B), np.zeros_like(model.W)]
    val = task.batch(np.random.default_rng(3000), 1024)
    vq, vx = task.inputs(val, content_only)
    history = []
    best = float("inf")
    best_params = None
    for step in range(1, steps + 1):
        batch = task.batch(rng, batch_size)
        loss, grads = model.loss_grad(*task.inputs(batch, content_only), batch["y"])
        for p, g, m, v in zip((model.B, model.W), grads, moments, variances):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            p -= 0.02 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if step == 1 or step % 200 == 0 or step == steps:
            vm = metrics(model.forward(vq, vx)[0], val["y"])
            history.append({"step": step, "train_batch_mse": loss, **vm})
            if vm["nmse"] < best:
                best, best_params = vm["nmse"], (model.B.copy(), model.W.copy(), step)
            print(f"seed={seed} content_only={content_only} step={step} val_nmse={vm['nmse']:.5f} accuracy={vm['signed_pair_accuracy']:.3f}", flush=True)
    model.B[:], model.W[:] = best_params[:2]
    return model, history, best_params[2]


def editing_batch(task, rng, size):
    batch = task.batch(rng, size * 2, parity=1, heldout=False)
    # ID 0 -> ID 1 should leave the negative variable distinct on edited rows.
    keep = (batch["pairs"][:, 0] != 0) | (batch["pairs"][:, 1] >= 2)
    idx = np.flatnonzero(keep)[:size]
    assert len(idx) == size
    return {k: v[idx] for k, v in batch.items()}


def evaluate(task, model, content_only=False):
    results = {}
    for name, parity, heldout in [("iid", 0, False), ("heldout_bindings", 1, False),
                                  ("heldout_queries", 0, True), ("joint_heldout", 1, True)]:
        b = task.batch(np.random.default_rng(4000 + parity * 10 + heldout), 4096, parity, heldout)
        results[name] = metrics(model.forward(*task.inputs(b, content_only))[0], b["y"])
    if content_only:
        return results
    b = task.batch(np.random.default_rng(4020), 2048, 1, True)
    q, x = task.inputs(b)
    clean, cache = model.forward(q, x)
    permuted, _ = model.forward(q, x[:, ::-1])
    results["source_order_max_abs_change"] = float(np.max(np.abs(clean - permuted)))
    assert results["source_order_max_abs_change"] < 1e-10
    swapq = b["q"].copy()
    swapq[:, task.p], swapq[:, task.m] = b["q"][:, task.m], b["q"][:, task.p]
    results["reversed_query"] = metrics(model.forward(swapq @ task.decoder.T, x)[0], -b["y"])
    swapz = b["z"].copy()
    for row, (i, j) in enumerate(b["pairs"]):
        swapz[row, :, [i, j]] = b["z"][row, :, [j, i]]
    results["swapped_binding_tags"] = metrics(model.forward(q, swapz @ task.decoder.T)[0], -b["y"])
    BF = task.decoder.T @ model.B @ task.decoder / model.scale
    binding_s = np.einsum("bi,hij,bnj->bhn", b["q"], BF[:, :, task.k], b["z"][:, :, task.k])
    content_s = np.einsum("bi,hij,bnj->bhn", b["q"], BF[:, :, task.c], b["z"][:, :, task.c])
    results["score_reconstruction_max_abs_error"] = float(np.max(np.abs(binding_s + content_s - cache["scores"])))
    assert results["score_reconstruction_max_abs_error"] < 1e-10
    results["binding_only_QK"] = metrics(model.forward(q, x, scores=binding_s)[0], b["y"])
    results["content_only_QK"] = metrics(model.forward(q, x, scores=content_s)[0], b["y"])
    WF = task.decoder.T @ model.W
    results["OV_content_scalar_gains"] = [float(np.trace(w[task.c]) / task.v) for w in WF]
    results["OV_content_distance_from_scalar_identity"] = [
        float(np.linalg.norm(w[task.c] - np.trace(w[task.c]) / task.v * np.eye(task.v)) /
              max(np.linalg.norm(w[task.c]), 1e-12)) for w in WF]
    results["QK_binding_row_contrasts"] = {}
    for role, offset in [("positive", task.n), ("negative", 2 * task.n)]:
        blocks = BF[:, offset:offset + task.n, task.k]
        results["QK_binding_row_contrasts"][role] = [
            float(np.mean(np.diag(w) - (w.sum(axis=1) - np.diag(w)) / (task.n - 1))) for w in blocks]

    eb = editing_batch(task, np.random.default_rng(5000), 4096)
    cf, gate = task.counterfactual(eb)
    eq, ex = task.inputs(eb)
    baseline, ecache = model.forward(eq, ex)
    feature_cf = model.forward(*task.inputs(cf))[0]
    patchedBF = BF.copy()
    patchedBF[:, task.n, task.k] = BF[:, task.n + 1, task.k]
    delta_s = np.einsum("bi,hij,bnj->bhn", eb["q"], patchedBF - BF, eb["z"])
    pair_scores = ecache["scores"] + delta_s
    pair_pred = model.forward(eq, ex, B=task.decoder @ (patchedBF * model.scale) @ task.decoder.T)[0]
    map_pred = model.forward(eq, ex, scores=pair_scores)[0]
    results["matched_pair_map_max_abs_difference"] = float(np.max(np.abs(pair_pred - map_pred)))
    assert results["matched_pair_map_max_abs_difference"] < 1e-10
    cal = editing_batch(task, np.random.default_rng(5001), 4096)
    calcf, calgate = task.counterfactual(cal)
    calbase = model.forward(*task.inputs(cal))[0]
    dom = np.mean(calcf["y"][calgate] - cal["y"][calgate], axis=0)
    desired = calcf["y"][calgate] - calbase[calgate]
    alpha = float(np.sum(desired * dom) / max(len(desired) * np.dot(dom, dom), 1e-15))
    dompred = baseline + gate[:, None] * alpha * dom
    results["edit_calibration"] = {"DoM_norm": float(np.linalg.norm(dom)), "DoM_gain": alpha}
    results["editing"] = {}
    for name, pred in [("no_edit", baseline), ("FRA_binding_pairs", pair_pred),
                       ("single_Q_feature_swap", feature_cf), ("matched_token_score_edit", map_pred),
                       ("gated_output_DoM", dompred)]:
        desired_delta = cf["y"][gate] - eb["y"][gate]
        change = pred[gate] - baseline[gate]
        progress = float(np.sum(change * desired_delta) / np.sum(desired_delta ** 2))
        rows = np.flatnonzero(gate)
        negative_ids = eb["bound"][rows, eb["pairs"][rows, 1]]
        protected_coordinates = desired_delta == 0
        results["editing"][name] = {
            "target_counterfactual_nmse": metrics(pred[gate], cf["y"][gate])["nmse"],
            "target_signed_pair_accuracy": metrics(pred[gate], cf["y"][gate])["signed_pair_accuracy"],
            "target_progress": progress,
            "protected_output_change_nmse": float(np.mean(np.sum((pred[~gate] - baseline[~gate]) ** 2, axis=-1)) / 2),
            "protected_output_max_abs_change": float(np.max(np.abs(pred[~gate] - baseline[~gate]))),
            "negative_operand_mean_abs_change": float(np.mean(np.abs(pred[rows, negative_ids] - baseline[rows, negative_ids]))),
            "target_protected_coordinates_rms_change": float(np.sqrt(np.mean(np.sum((change * protected_coordinates) ** 2, axis=-1)))),
        }
    return results


def plot_and_report(out, results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    seeds = results["trained"]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.10,
                        subplot_titles=["Learning signed variable retrieval", "Which QK terms are used?", "Rebind positive variable 0 to 1"])
    for run in seeds:
        fig.add_trace(go.Scatter(x=[r["step"] for r in run["history"]], y=[r["nmse"] for r in run["history"]],
                                name=f"Binding tags, seed {run['seed']}", mode="lines+markers"), row=1, col=1)
    control = results["content_only_control"]
    fig.add_trace(go.Scatter(x=[r["step"] for r in control["history"]], y=[r["nmse"] for r in control["history"]],
                            line={"color": "black", "dash": "dash"}, name="No binding tags"), row=1, col=1)
    fig.update_xaxes(title_text="Training steps", row=1, col=1)
    fig.update_yaxes(title_text="Validation normalized MSE", type="log", row=1, col=1)
    labels = ["Joint held-out", "Binding QK<br>terms only", "Content QK<br>terms only"]
    keys = ["joint_heldout", "binding_only_QK", "content_only_QK"]
    values = np.array([[r["evaluation"][k]["signed_pair_accuracy"] for k in keys] for r in seeds])
    fig.add_trace(go.Bar(x=labels, y=values.mean(0), marker_color=["#377eb8", "#4daf4a", "#e41a1c"],
                        opacity=.7, showlegend=False, name="Seed mean"), row=1, col=2)
    for run, row in zip(seeds, values):
        fig.add_trace(go.Scatter(x=labels, y=row, mode="markers", marker={"color": "black", "size": 6},
                                name=f"Seed {run['seed']}", showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="Both signed fillers correct", range=[0, 1.05], row=1, col=2)
    methods = ["no_edit", "FRA_binding_pairs", "single_Q_feature_swap", "matched_token_score_edit", "gated_output_DoM"]
    vals = np.array([[r["evaluation"]["editing"][m]["target_counterfactual_nmse"] for m in methods] for r in seeds])
    labels = ["No edit", "FRA<br>pairs", "Q feature<br>swap", "Matched<br>map", "Gated<br>DoM"]
    fig.add_trace(go.Bar(x=labels, y=vals.mean(0), marker_color=["gray", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
                        opacity=.7, showlegend=False, name="Seed mean"), row=1, col=3)
    for run, row in zip(seeds, vals):
        fig.add_trace(go.Scatter(x=labels, y=row, mode="markers", marker={"color": "black", "size": 6},
                                name=f"Seed {run['seed']}", showlegend=False), row=1, col=3)
    fig.update_yaxes(title_text="Counterfactual normalized MSE", row=1, col=3)
    fig.update_layout(title="Supplied binding features · two learned softmax heads · CPU pilot",
                      template="plotly_white", height=540, width=1500, margin={"t": 110, "b": 130},
                      legend={"orientation": "h", "y": -.25, "x": 0}, font={"family": "Arial", "size": 12})
    fig.write_html(str(out / "binding_pilot.html"), include_plotlyjs=True)
    lines = ["# Variable-binding pilot results", "", "This is a supplied-feature, two-head synthetic pilot. It does not test SAE recovery, discovery of binding features from raw tokens, or an FRA advantage over arbitrary map edits.", "", "## Held-out performance", "", "| Seed | IID accuracy | Unseen bindings | Unseen queries | Joint held-out | Joint NMSE |", "|---|---:|---:|---:|---:|---:|"]
    for r in seeds:
        e = r["evaluation"]
        lines.append(f"| {r['seed']} | {e['iid']['signed_pair_accuracy']:.4f} | {e['heldout_bindings']['signed_pair_accuracy']:.4f} | {e['heldout_queries']['signed_pair_accuracy']:.4f} | {e['joint_heldout']['signed_pair_accuracy']:.4f} | {e['joint_heldout']['nmse']:.6f} |")
    lines += ["", "## Mechanistic checks", "", "| Seed | Binding-only QK accuracy | Content-only QK accuracy | Reversed-query accuracy | Binding-swap accuracy |", "|---|---:|---:|---:|---:|"]
    for r in seeds:
        e = r["evaluation"]
        lines.append(f"| {r['seed']} | {e['binding_only_QK']['signed_pair_accuracy']:.4f} | {e['content_only_QK']['signed_pair_accuracy']:.4f} | {e['reversed_query']['signed_pair_accuracy']:.4f} | {e['swapped_binding_tags']['signed_pair_accuracy']:.4f} |")
    lines += ["", "## Rebinding edits", "", "Numbers below are means across the trained seeds. Protected-query zero damage is imposed by the common edit gate, not an empirical selectivity advantage. Negative-operand drift measures collateral within edited queries and was added after the initial run.", "", "| Method | Target counterfactual NMSE | Target progress | Negative-operand mean absolute drift | Protected-query output-change NMSE |", "|---|---:|---:|---:|---:|"]
    for m in methods:
        es = [r["evaluation"]["editing"][m] for r in seeds]
        lines.append(f"| {m} | {np.mean([e['target_counterfactual_nmse'] for e in es]):.6f} | {np.mean([e['target_progress'] for e in es]):.6f} | {np.mean([e['negative_operand_mean_abs_change'] for e in es]):.6f} | {np.mean([e['protected_output_change_nmse'] for e in es]):.3g} |")
    lines += ["", f"Content-only control joint held-out NMSE: {control['evaluation']['joint_heldout']['nmse']:.6f}; signed-pair accuracy: {control['evaluation']['joint_heldout']['signed_pair_accuracy']:.4f}. Population-optimal NMSE without binding information is 1.", "", f"Maximum FRA/matched-map difference: {max(r['evaluation']['matched_pair_map_max_abs_difference'] for r in seeds):.3g}.", "", "The DoM baseline is a fixed context-independent output direction with the same query gate. Its failure alone is not evidence for a distinctive FRA benefit. The single Q-feature swap is the stronger comparator.", "", "[Interactive pilot figure](binding_pilot.html)", "", "Full numerical results, configuration, fixed seeds, and numerical checks are in metrics.json; learned arrays are in seed_*.npz."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    task = Task()
    checks = validate_math(task)
    print("Mathematical checks:", checks, flush=True)
    results = {"config": {"steps": args.steps, "batch_size": args.batch_size, "seeds": args.seeds,
                            "variables": task.n, "fillers": task.v, "heads": 2, "decoder_seed": 123,
                            "optimizer": "Adam", "learning_rate": 0.02, "numpy": np.__version__,
                            "python": platform.python_version(), "device": "CPU",
                            "checkpoint_selection": "minimum IID validation NMSE at logged checkpoints"},
               "checks": checks, "trained": []}
    for seed in args.seeds:
        model, history, selected_step = train(task, seed, args.steps, args.batch_size)
        evaluation = evaluate(task, model)
        results["trained"].append({"seed": seed, "history": history, "selected_step": selected_step, "evaluation": evaluation})
        np.savez(args.out / f"seed_{seed}.npz", B=model.B, W=model.W, decoder=task.decoder)
        print("Evaluation seed", seed, json.dumps({k: evaluation[k] for k in ["iid", "joint_heldout", "binding_only_QK", "content_only_QK"]}), flush=True)
    control, history, selected_step = train(task, 0, args.steps, args.batch_size, content_only=True)
    results["content_only_control"] = {"seed": 0, "history": history, "selected_step": selected_step,
                                       "evaluation": evaluate(task, control, content_only=True)}
    np.savez(args.out / "content_only_control.npz", B=control.B, W=control.W, decoder=task.decoder)
    results["elapsed_seconds"] = time.time() - start
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_and_report(args.out, results)
    print("Completed:", args.out, "seconds:", results["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
