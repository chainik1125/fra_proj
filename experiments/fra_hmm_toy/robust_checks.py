"""Robustness battery for the FRA-OV gain-tuned null (use removed, concept kept).

Against out/phase2_L0 checkpoints, on-distribution eval (conc 10):
  1. Feature-set dependence: gain curves for omega_top1 / top2 / top4 —
     does a thinner channel still null at higher gain?
  2. Position-split leakage: early (t 8-127) vs late (t 128-254) RF/tracking —
     an additive counter-gain would leak where the signal is stronger.
  3. Nonlinear presence: 2-layer MLP probe vs ridge on clean and on the
     c=4 null point — "concept kept" must not be a linearity artifact.
  4. Gain sensitivity near the null: finite-difference dRF/dc.
Writes out/robust_checks.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra_cut import FRAToolkit, Intervention
from metrics import factorized_ce, ridge_probe_r2, tracking_r2
from mixture_data import make_dataset
from toy_model import TopKSAE, build_transformer

OUT = Path(__file__).resolve().parent / "out"
SEED, N_EVAL, SEQ_LEN, T_MIN, CHUNK = 42, 500, 256, 8, 25
SPLIT_COL = 120  # position 128 in absolute terms (t_min offset 8)

torch.set_grad_enabled(False)
prev = json.load(open(OUT / "phase2_L0" / "results.json"))
sets = {k: torch.tensor(v, dtype=torch.long) for k, v in prev["cut_sets"].items()}

eval_ds = make_dataset(SEED + 1, N_EVAL, SEQ_LEN, 10.0)
block_map = eval_ds.block_of_token
K = eval_ds.config.K

model = build_transformer(eval_ds.config.V_total, n_ctx=SEQ_LEN, seed=SEED, device="cpu")
model.load_state_dict(torch.load(OUT / "phase2_L0" / "transformer.pt", map_location="cpu"))
model.eval()
sae = TopKSAE(d_in=64, d_sae=64, k=4)
sae.load_state_dict(torch.load(OUT / "phase2_L0" / "sae.pt", map_location="cpu"))
tk = FRAToolkit(model, sae, l_sae=0)

# split-specific Bayes anchors (block factor), computed per position bucket
tokens = eval_ds.tokens
post = eval_ds.posterior_omegas
tgt_block = block_map[tokens[:, 1:]]
omega_pred = post[:, :-1]
bayes_blk_pt = -omega_pred.gather(-1, tgt_block.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12).log()[:, T_MIN:]
prior = torch.tensor(eval_ds.config.omega)
prior_blk_pt = -prior[tgt_block].clamp(min=1e-12).log()[:, T_MIN:]
omega_pred_sl = omega_pred[:, T_MIN:]

BUCKETS = {"all": slice(None), "early": slice(0, SPLIT_COL), "late": slice(SPLIT_COL, None)}
anch = {
    b: {"bayes": float(bayes_blk_pt[:, sl].mean()), "prior": float(prior_blk_pt[:, sl].mean())}
    for b, sl in BUCKETS.items()
}

GRIDS = {
    "omega_top4": [3.0, 3.5, 4.0, 4.5, 5.0],
    "omega_top2": [3.0, 4.5, 6.0, 7.5, 9.0],
    "omega_top1": [4.0, 6.0, 8.0, 12.0, 16.0],
}

ivs = [Intervention("clean", "none")]
for sname, cs in GRIDS.items():
    for c in cs:
        ivs.append(Intervention(f"ov:{sname}:c{c}", "fra_ov", S=sets[sname], strength=c))

resid2_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
results: dict = {}
clean_blk = {}
mlp_targets = {}
n_chunks = (N_EVAL + CHUNK - 1) // CHUNK

for iv in ivs:
    blk_pts, bm_all, resid_all = [], [], []
    wce_sum = wce_n = 0.0
    for ci in range(n_chunks):
        chunk = eval_ds.tokens[ci * CHUNK : (ci + 1) * CHUNK]
        cctx = tk.clean_context(chunk)
        hooks = iv.build_hooks(tk, cctx)
        grab = {}

        def grab_hook(act, hook):
            grab["r"] = act.detach()
            return act

        logits = model.run_with_hooks(chunk, fwd_hooks=hooks + [(resid2_hook, grab_hook)], return_type="logits")
        ce = factorized_ce(logits, chunk, block_map, t_min=T_MIN)
        blk_pts.append(ce["block_ce"])
        wce_sum += float(ce["within_ce"].sum()); wce_n += ce["within_ce"].numel()
        bm_all.append(ce["block_mass"])
        resid_all.append(grab["r"][:, T_MIN:-1])

    blk = torch.cat(blk_pts)          # (N, 247)
    bm = torch.cat(bm_all)            # (N, 247, K)
    row = {"within_ce": wce_sum / wce_n}
    for b, sl in BUCKETS.items():
        bce = float(blk[:, sl].mean())
        if iv.name == "clean":
            clean_blk[b] = bce
        row[f"block_ce_{b}"] = bce
        row[f"rf_{b}"] = (bce - clean_blk[b]) / (anch[b]["prior"] - anch[b]["bayes"])
        row[f"track_{b}"] = tracking_r2(bm[:, sl], omega_pred_sl[:, sl])
    xall = torch.cat(resid_all)
    row["probe_r2_linear"] = ridge_probe_r2(
        xall.reshape(-1, 64).numpy().astype(np.float64),
        omega_pred_sl[:, : xall.shape[1]].reshape(-1, K).numpy().astype(np.float64),
        n_sequences=N_EVAL, per_seq=xall.shape[1], seed=SEED,
    )
    if iv.name in ("clean", "ov:omega_top4:c4.0"):
        mlp_targets[iv.name] = xall.clone()
    results[iv.name] = row
    print(f"{iv.name:22s} RF all/early/late {row['rf_all']:+.3f}/{row['rf_early']:+.3f}/{row['rf_late']:+.3f}  "
          f"track {row['track_all']:+.3f}/{row['track_early']:+.3f}/{row['track_late']:+.3f}  "
          f"probe {row['probe_r2_linear']:+.3f}", flush=True)

# ── MLP probe (nonlinear presence) ──
def mlp_probe_r2(x: torch.Tensor, y: torch.Tensor, seed: int = 0) -> float:
    with torch.enable_grad():
        g = np.random.default_rng(seed)
        n_seq, per_seq, D = x.shape
        perm = g.permutation(n_seq)
        tr = torch.tensor(perm[: int(0.8 * n_seq)])
        te = torch.tensor(perm[int(0.8 * n_seq):])
        xtr = x[tr].reshape(-1, D).float(); ytr = y[tr].reshape(-1, K).float()
        xte = x[te].reshape(-1, D).float(); yte = y[te].reshape(-1, K).float()
        mu, sd = xtr.mean(0), xtr.std(0).clamp(min=1e-6)
        xtr = (xtr - mu) / sd; xte = (xte - mu) / sd
        net = torch.nn.Sequential(
            torch.nn.Linear(D, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, K),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        gen = torch.Generator().manual_seed(seed)
        for step in range(600):
            idx = torch.randint(xtr.shape[0], (4096,), generator=gen)
            loss = (net(xtr[idx]) - ytr[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            pred = net(xte)
        sse = (yte - pred).pow(2).sum(0)
        sst = (yte - yte.mean(0)).pow(2).sum(0)
        return float((1 - sse / sst.clamp(min=1e-12)).mean())

y_all = omega_pred_sl[:, : mlp_targets["clean"].shape[1]]
mlp = {name: mlp_probe_r2(x, y_all) for name, x in mlp_targets.items()}
print("MLP probe R2:", {k: round(v, 3) for k, v in mlp.items()}, flush=True)

# gain sensitivity near the top4 null
cs = GRIDS["omega_top4"]
rfs = [results[f"ov:omega_top4:c{c}"]["rf_all"] for c in cs]
drf_dc = float(np.gradient(np.array(rfs), np.array(cs))[cs.index(4.0)])

out = {"anchors_split": anch, "results": results, "mlp_probe_r2": mlp,
       "drf_dc_at_null_top4": drf_dc}
(OUT / "robust_checks.json").write_text(json.dumps(out, indent=2))
print(f"dRF/dc at null (top4): {drf_dc:.3f}", flush=True)
print("saved", OUT / "robust_checks.json", flush=True)
