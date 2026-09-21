"""Runner: can FRA cut 'which component' out of the mixture-HMM toy better
than SAE feature ablation?

Pipeline: data -> transformer -> TopK SAE @ blocks.1.hook_resid_post ->
feature ID (train data only) -> intervention grid (SAE cut / proj cut /
FRA-QK / FRA-OV / combined / random controls) -> exact removal-vs-collateral
metrics against analytic Bayes anchors + mechanistic QK mass decomposition.

Usage:
  python run_toy.py --out out/main            # full run
  python run_toy.py --out out/smoke --smoke   # quick correctness pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra_cut import FRAToolkit, Intervention
from metrics import bayes_anchors, factorized_ce, ridge_probe_r2, summarize
from mixture_data import make_dataset
from toy_model import (
    TopKSAE,
    build_transformer,
    extract_activations,
    identify_features,
    train_topk_sae,
    train_transformer,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=256, help="eval length == n_ctx == crop length")
    p.add_argument("--train-seq-len", type=int, default=768,
                   help="train sequences are longer; random crops of seq-len are the augmentation "
                        "that prevents memorization (the exp03 recipe's accidental virtue)")
    p.add_argument("--concentration", type=float, default=10.0)
    p.add_argument("--transformer-steps", type=int, default=2500)
    p.add_argument("--sae-steps", type=int, default=2000)
    p.add_argument("--dict-size", type=int, default=64)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--sae-layer", type=int, default=1)
    p.add_argument("--t-min", type=int, default=8)
    p.add_argument("--eval-chunk", type=int, default=25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def pick_device(arg):
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_top_per_component(r2: np.ndarray, m: int) -> list[int]:
    """r2: (d_sae, K). Top-m latents per component by that component's R²."""
    S: list[int] = []
    for c in range(r2.shape[1]):
        order = np.argsort(-r2[:, c])
        for i in order[:m]:
            if int(i) not in S:
                S.append(int(i))
    return S


def main():
    args = get_args()
    if args.smoke:
        args.n_train, args.n_eval = 96, 48
        args.seq_len, args.train_seq_len = 64, 128
        args.transformer_steps, args.sae_steps = 60, 120
        args.t_min = 4
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={device}  out={args.out}", flush=True)

    # ── data ──
    train_ds = make_dataset(args.seed, args.n_train, args.train_seq_len, args.concentration)
    eval_ds = make_dataset(args.seed + 1, args.n_eval, args.seq_len, args.concentration)
    V = train_ds.config.V_total
    K = train_ds.config.K
    block_map = train_ds.block_of_token

    # ── transformer ──
    tf_path = args.out / "transformer.pt"
    model = build_transformer(V, n_ctx=args.seq_len, seed=args.seed, device=device)
    if tf_path.exists() and not args.force:
        model.load_state_dict(torch.load(tf_path, map_location=device))
        print("loaded transformer checkpoint", flush=True)
    else:
        losses = train_transformer(
            model, train_ds.tokens, n_steps=args.transformer_steps,
            context_len=args.seq_len, seed=args.seed,
        )
        torch.save(model.state_dict(), tf_path)
        print(f"transformer final loss {losses[-1]:.4f}", flush=True)
    model.eval()
    torch.set_grad_enabled(False)

    hook_name = f"blocks.{args.sae_layer}.hook_resid_post"

    # ── SAE ──
    sae_path = args.out / "sae.pt"
    sae = TopKSAE(d_in=model.cfg.d_model, d_sae=args.dict_size, k=args.k).to(device)
    train_tokens_ctx = train_ds.tokens[:, : args.seq_len]  # n_ctx-length prefix
    train_acts = extract_activations(model, train_tokens_ctx, hook_name)
    flat_train_acts = train_acts.reshape(-1, train_acts.shape[-1])
    if sae_path.exists() and not args.force:
        sae.load_state_dict(torch.load(sae_path, map_location=device))
        print("loaded SAE checkpoint", flush=True)
    else:
        torch.set_grad_enabled(True)
        sae_losses = train_topk_sae(sae, flat_train_acts, n_steps=args.sae_steps, seed=args.seed)
        torch.set_grad_enabled(False)
        torch.save(sae.state_dict(), sae_path)
        print(f"sae final loss {sae_losses[-1]:.4f}", flush=True)

    with torch.no_grad():
        xh, _ = sae(flat_train_acts.to(device))
        fvu = float(((flat_train_acts.to(device) - xh) ** 2).sum() /
                    ((flat_train_acts.to(device) - flat_train_acts.to(device).mean(0)) ** 2).sum())
    print(f"SAE FVU on train acts: {fvu:.4f}", flush=True)

    # ── feature identification (TRAIN data only) ──
    from toy_model import encode_all
    z_train = encode_all(sae, flat_train_acts)
    token_block_train = block_map[train_tokens_ctx.reshape(-1)]
    feats = identify_features(
        z_train, train_ds.posterior_omegas[:, : args.seq_len], token_block_train,
        n_positions_skip=args.t_min, seq_len=args.seq_len,
    )
    r2o, r2b = feats["r2_omega"], feats["r2_block"]
    print("\ntop omega-tracking latents (latent, comp, R²):")
    for i in np.argsort(-feats["best_omega_r2"])[:8]:
        print(f"  {i:3d}  c{feats['best_omega_comp'][i]}  {feats['best_omega_r2'][i]:.3f}")
    print("top token-block latents (latent, comp, R²):")
    for i in np.argsort(-feats["best_block_r2"])[:8]:
        print(f"  {i:3d}  c{feats['best_block_comp'][i]}  {feats['best_block_r2'][i]:.3f}")

    S_sets = {
        "omega_top1": select_top_per_component(r2o, 1),
        "omega_top2": select_top_per_component(r2o, 2),
        "omega_top4": select_top_per_component(r2o, 4),
        "block_top2": select_top_per_component(r2b, 2),
        "block_top4": select_top_per_component(r2b, 4),
    }
    rng = np.random.default_rng(args.seed)
    S_sets["rand6_a"] = [int(i) for i in rng.choice(args.dict_size, size=len(S_sets["omega_top2"]), replace=False)]
    S_sets["rand6_b"] = [int(i) for i in rng.choice(args.dict_size, size=len(S_sets["omega_top2"]), replace=False)]
    print("\ncut sets:", {k: v for k, v in S_sets.items()}, flush=True)

    # probe directions for proj_cut (fit on train acts -> posterior omega)
    xtr = flat_train_acts.numpy().astype(np.float64)
    ytr = train_ds.posterior_omegas[:, : args.seq_len].reshape(-1, K).numpy().astype(np.float64)
    xm, ym_ = xtr.mean(0, keepdims=True), ytr.mean(0, keepdims=True)
    xc, yc = xtr - xm, ytr - ym_
    xtx = xc.T @ xc
    xtx.flat[:: xtx.shape[0] + 1] += 1e-4 * np.trace(xtx) / xtx.shape[0]
    beta = np.linalg.solve(xtx, xc.T @ yc)          # (D, K)
    Qdirs, _ = np.linalg.qr(beta)
    probe_dirs = torch.tensor(Qdirs, dtype=torch.float32, device=device)

    # ── toolkit + interventions ──
    tk = FRAToolkit(model, sae, l_sae=args.sae_layer)

    def S_t(name):
        return torch.tensor(S_sets[name], dtype=torch.long, device=device)

    ivs: list[Intervention] = [Intervention("clean", "none")]
    for sname in ["omega_top1", "omega_top2", "omega_top4", "block_top2", "block_top4", "rand6_a", "rand6_b"]:
        for a in ([1.0] if sname.startswith("rand") else [0.5, 1.0]):
            ivs.append(Intervention(f"sae:{sname}:a{a}", "sae_cut", S=S_t(sname), strength=a))
    for a in [0.5, 1.0]:
        ivs.append(Intervention(f"proj:probe3:a{a}", "proj_cut", dirs=probe_dirs, strength=a))
    for sname in ["omega_top2", "omega_top4", "block_top2", "block_top4", "rand6_a"]:
        for side in ["key", "query", "either"]:
            for c in [1.0, 2.0]:
                ivs.append(Intervention(f"qk:{sname}:{side}:c{c}", "fra_qk", S=S_t(sname), side=side, strength=c))
    for sname in ["omega_top2", "omega_top4", "block_top2", "block_top4", "rand6_a"]:
        for c in [1.0, 2.0]:
            ivs.append(Intervention(f"ov:{sname}:c{c}", "fra_ov", S=S_t(sname), strength=c))
    for sname in ["omega_top2", "omega_top4", "block_top4"]:
        ivs.append(Intervention(f"qkov:{sname}:either:c1.0", "fra_qk_ov", S=S_t(sname), side="either", strength=1.0))
    print(f"\n{len(ivs)} interventions", flush=True)

    # ── anchors ──
    anchors = bayes_anchors(eval_ds, slice(None), t_min=args.t_min)
    print(
        f"anchors: bayes block CE {anchors['bayes_block_ce']:.4f} | prior block CE {anchors['prior_block_ce']:.4f}"
        f" | bayes within CE {anchors['bayes_within_ce']:.4f} | uniform within {anchors['uniform_within_ce']:.4f}",
        flush=True,
    )

    # ── verification + mechanistic readout on a subsample ──
    sub = eval_ds.tokens[:8].to(device)
    ctx = tk.clean_context(sub)
    ver = tk.verify_decomposition(ctx)
    print(f"FRA decomposition exactness: max_abs_err={ver['max_abs_err']:.2e} ok={bool(ver['ok'])}", flush=True)
    mass = tk.qk_mass_by_type(ctx)
    mech = {
        "verify": ver,
        "types": mass["types"],
        "qk_mass_raw": mass["raw"].tolist(),
        "qk_mass_pattern_weighted": mass["pattern_weighted"].tolist(),
    }
    print("QK |score| mass by (q-type x k-type), pattern-weighted, per head:")
    pw = mass["pattern_weighted"]
    for h in range(pw.shape[2]):
        print(f"  head {h}:")
        for a, ta in enumerate(mass["types"]):
            row = "  ".join(f"{ta}x{tb}={pw[a, b, h]:.3f}" for b, tb in enumerate(mass["types"]))
            print(f"    {row}")

    # ── evaluation loop ──
    eval_tokens = eval_ds.tokens
    n_chunks = (args.n_eval + args.eval_chunk - 1) // args.eval_chunk
    results = {}
    clean_block_ce = clean_within_ce = None
    resid2_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    for iv in ivs:
        agg = {"block_ce_sum": 0.0, "within_ce_sum": 0.0, "n": 0}
        bm_all, resid_all = [], []
        for ci in range(n_chunks):
            chunk = eval_tokens[ci * args.eval_chunk : (ci + 1) * args.eval_chunk].to(device)
            cctx = tk.clean_context(chunk)
            hooks = iv.build_hooks(tk, cctx)
            grab = {}

            def grab_hook(act, hook):
                grab["resid2"] = act.detach()
                return act

            logits = model.run_with_hooks(
                chunk, fwd_hooks=hooks + [(resid2_hook, grab_hook)], return_type="logits",
            )
            ce = factorized_ce(logits.cpu(), chunk.cpu(), block_map, t_min=args.t_min)
            agg["block_ce_sum"] += float(ce["block_ce"].sum())
            agg["within_ce_sum"] += float(ce["within_ce"].sum())
            agg["n"] += ce["block_ce"].numel()
            bm_all.append(ce["block_mass"])
            resid_all.append(grab["resid2"][:, args.t_min : -1].cpu())

        block_mass = torch.cat(bm_all, dim=0)
        block_ce = agg["block_ce_sum"] / agg["n"]
        within_ce = agg["within_ce_sum"] / agg["n"]
        if iv.name == "clean":
            clean_block_ce, clean_within_ce = block_ce, within_ce

        res = summarize(
            {"block_ce": torch.tensor([[block_ce]]), "within_ce": torch.tensor([[within_ce]]),
             "block_mass": block_mass, "tgt_block": None},
            anchors, clean_block_ce, clean_within_ce,
        )
        res["block_ce"], res["within_ce"] = block_ce, within_ce

        # adversarial probe on last-layer resid
        xall = torch.cat(resid_all, dim=0)
        per_seq = xall.shape[1]
        res["probe_r2_resid_last"] = ridge_probe_r2(
            xall.reshape(-1, xall.shape[-1]).numpy().astype(np.float64),
            anchors["omega_pred"][:, : per_seq].reshape(-1, K).numpy().astype(np.float64),
            n_sequences=args.n_eval, per_seq=per_seq, seed=args.seed,
        )
        results[iv.name] = res
        print(
            f"{iv.name:32s} blockCE {block_ce:.4f}  withinCE {within_ce:.4f}  "
            f"track {res['tracking_r2']:+.3f}  probe {res['probe_r2_resid_last']:+.3f}"
            + (f"  RF {res.get('removal_frac', 0):+.3f} CF {res.get('collateral_frac', 0):+.3f}"
               if "removal_frac" in res else ""),
            flush=True,
        )

    out = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "anchors": {k: v for k, v in anchors.items() if not isinstance(v, torch.Tensor)},
        "sae_fvu": fvu,
        "cut_sets": S_sets,
        "feature_tables": {
            "best_omega_r2": feats["best_omega_r2"].tolist(),
            "best_omega_comp": feats["best_omega_comp"].tolist(),
            "best_block_r2": feats["best_block_r2"].tolist(),
            "best_block_comp": feats["best_block_comp"].tolist(),
        },
        "mechanistic": mech,
        "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
