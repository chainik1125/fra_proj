"""Strength sweep of the presence-preserving corner: FRA-OV @ L0 (omega feats)
vs presence-matched SAE cuts. Where does the horizontal line end?

Reuses out/phase2_L0/{transformer.pt,sae.pt}. Sweeps:
  - fra_ov, S = omega_top4 (from phase2_L0 results), c in a fine grid
  - sae_cut, same S, small alphas (the presence-matched comparison curve)
Reports RF / CF / probe / tracking per point -> out/ov_sweep.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra_cut import FRAToolkit, Intervention
from metrics import bayes_anchors, factorized_ce, ridge_probe_r2, summarize
from mixture_data import make_dataset
from toy_model import TopKSAE, build_transformer

OUT = Path(__file__).resolve().parent / "out"
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--conc", type=float, default=10.0)
_ap.add_argument("--out-name", type=str, default="ov_sweep.json")
_ap.add_argument("--dir", type=str, default="phase2_L0")
_ap.add_argument("--seed", type=int, default=42)
_args = _ap.parse_args()
SEED, N_EVAL, SEQ_LEN, T_MIN, CHUNK = _args.seed, 500, 256, 8, 25

torch.set_grad_enabled(False)
prev = json.load(open(OUT / _args.dir / "results.json"))
S_omega = prev["cut_sets"]["omega_top4"]

eval_ds = make_dataset(SEED + 1, N_EVAL, SEQ_LEN, _args.conc)
block_map = eval_ds.block_of_token
K = eval_ds.config.K

model = build_transformer(eval_ds.config.V_total, n_ctx=SEQ_LEN, seed=SEED, device="cpu")
model.load_state_dict(torch.load(OUT / _args.dir / "transformer.pt", map_location="cpu"))
model.eval()
sae = TopKSAE(d_in=64, d_sae=64, k=4)
sae.load_state_dict(torch.load(OUT / _args.dir / "sae.pt", map_location="cpu"))

tk = FRAToolkit(model, sae, l_sae=0)
anchors = bayes_anchors(eval_ds, slice(None), t_min=T_MIN)
S = torch.tensor(S_omega, dtype=torch.long)

ivs = [Intervention("clean", "none")]
for c in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
    ivs.append(Intervention(f"ov:omega_top4:c{c}", "fra_ov", S=S, strength=c))
for a in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    ivs.append(Intervention(f"sae:omega_top4:a{a}", "sae_cut", S=S, strength=a))

resid2_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
results = {}
clean_block_ce = clean_within_ce = None
n_chunks = (N_EVAL + CHUNK - 1) // CHUNK

for iv in ivs:
    agg_b = agg_w = 0.0
    n = 0
    bm_all, resid_all = [], []
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
        agg_b += float(ce["block_ce"].sum())
        agg_w += float(ce["within_ce"].sum())
        n += ce["block_ce"].numel()
        bm_all.append(ce["block_mass"])
        resid_all.append(grab["r"][:, T_MIN:-1])

    block_ce, within_ce = agg_b / n, agg_w / n
    if iv.name == "clean":
        clean_block_ce, clean_within_ce = block_ce, within_ce
    res = summarize(
        {"block_ce": torch.tensor([[block_ce]]), "within_ce": torch.tensor([[within_ce]]),
         "block_mass": torch.cat(bm_all), "tgt_block": None},
        anchors, clean_block_ce, clean_within_ce,
    )
    res["block_ce"], res["within_ce"] = block_ce, within_ce
    xall = torch.cat(resid_all)
    per_seq = xall.shape[1]
    res["probe_r2"] = ridge_probe_r2(
        xall.reshape(-1, 64).numpy().astype(np.float64),
        anchors["omega_pred"][:, :per_seq].reshape(-1, K).numpy().astype(np.float64),
        n_sequences=N_EVAL, per_seq=per_seq, seed=SEED,
    )
    results[iv.name] = res
    print(f"{iv.name:24s} RF {res.get('removal_frac', 0):+.3f}  CF {res.get('collateral_frac', 0):+.3f}  "
          f"probe {res['probe_r2']:+.3f}  track {res['tracking_r2']:+.3f}", flush=True)

(OUT / _args.out_name).write_text(json.dumps(results, indent=2))
print("saved", OUT / _args.out_name, flush=True)
