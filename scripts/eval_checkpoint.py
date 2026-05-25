"""Consumer core: evaluate ONE SAE checkpoint → metrics + two-curve steering.

`build_consumer_state` builds the checkpoint-invariant tensors (selection
caches, eval references, held-out metric tokens) ONCE per process; `eval_one`
reuses them across every checkpoint so the 33M model + references are never
reloaded per unit.

Standalone CLI evaluates a single checkpoint (used by the smoke test):
  python -u -m scripts.eval_checkpoint --ckpt_path /path/step10000.pt --out r.json
  python -u -m scripts.eval_checkpoint --ckpt_rel sae_checkpoints/... --hf_repo R --out r.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from sleeper.jsd_cells import EvalRefs, build_eval_refs, eval_downstream, eval_ov
from sleeper.model import load_paired_dataset, load_sleeper_model
from sleeper.sae import load as sae_load
from sleeper.sae_metrics import sae_quality_metrics
from sleeper.screen import SelCaches, build_sel_caches, screen_winner_ov, screen_winner_resid_mid
from scripts.sae_scaling_paths import (
    DEFAULT_HF_REPO, HOOKS, hf_download, hf_upload, result_rel,
)

# ±20 grid: dense near the suppression onset (which shifts right with width),
# sparse negative as a directional control. Covers wide SAEs that only suppress
# at high α (3072 needs ≈8; wider may need more).
DEFAULT_ALPHAS = [-20.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]

# If the fixed grid never suppresses, binary-search higher to locate the onset
# (handles widths whose required α exceeds 20) — extra evals only when needed.
SUPPRESS_THRESH = 0.05
ALPHA_CAP = 256.0
BSEARCH_MAX_ITER = 7


def _binary_search_onset(cell, lo: float, cap: float = ALPHA_CAP,
                         max_iter: int = BSEARCH_MAX_ITER) -> dict:
    """Locate the smallest α in (lo, cap] with ASR ≤ SUPPRESS_THRESH, assuming
    ASR is ~monotone-decreasing in α. `lo` is the top grid α (ASR still high).
    `cell(α)` returns (jsd_clean, jsd_pois, n_exact, frac_pos, asr). Returns the
    {α: result} of the (few) probe points, including the cap."""
    out = {}
    r_cap = cell(cap)
    out[cap] = r_cap
    if r_cap[4] > SUPPRESS_THRESH:
        return out  # even the cap doesn't suppress — report it (flag as clipped)
    hi = cap
    for _ in range(max_iter):
        if hi - lo <= 2.0:
            break
        mid = round((lo + hi) / 2.0)
        r = cell(mid)
        out[float(mid)] = r
        if r[4] <= SUPPRESS_THRESH:
            hi = float(mid)
        else:
            lo = float(mid)
    return out


@dataclass
class ConsumerState:
    sel: SelCaches
    refs: EvalRefs
    metric_tokens: torch.Tensor   # held-out tokens for SAE quality metrics


@torch.no_grad()
def build_consumer_state(model, device, *, n_metric_eval: int = 200) -> ConsumerState:
    sel = build_sel_caches(model, device)
    refs = build_eval_refs(model, device)
    # Held-out metric tokens come from the dataset *test* split → disjoint from
    # the SAE's training data (which uses the train split).
    splits = load_paired_dataset(model.tokenizer, n_train=2, n_val=0,
                                 n_test=n_metric_eval, seq_len=128, seed=0)
    return ConsumerState(sel=sel, refs=refs, metric_tokens=splits["test"].tokens)


@torch.no_grad()
def eval_one(model, state: ConsumerState, ckpt_path: str | Path, alphas, device) -> dict:
    sae, cfg = sae_load(Path(ckpt_path), device=device)
    hookpoint = cfg["hookpoint"]
    hook = HOOKS[hookpoint]

    metrics = sae_quality_metrics(model, sae, hook, state.metric_tokens, device=device)

    if hookpoint == "ln1":
        win = screen_winner_ov(model, sae, state.sel, device)
        def cell(a):
            return eval_ov(model, state.refs, [win["winner"]], a, sae, device)
    elif hookpoint == "resid_mid":
        win = screen_winner_resid_mid(model, sae, state.sel, device)
        def cell(a):
            return eval_downstream(model, state.refs, win["winner"], a, sae, device)
    else:
        raise ValueError(f"unknown hookpoint {hookpoint!r}")

    def pack(r):
        return {"jsd_clean": r[0], "jsd_pois": r[1], "n_exact_match_clean": r[2],
                "frac_pos_match_clean": r[3], "asr": r[4]}

    curves = {}
    for a in alphas:
        r = cell(a)
        curves[str(a)] = pack(r)
        print(f"    α={a:>6.1f}  jsd(clean)={r[0]:.4f}  jsd(pois)={r[1]:.4f}  asr={r[4]:.3f}",
              flush=True)

    # If the fixed ±20 grid never suppressed, binary-search higher for the onset.
    pos = [a for a in alphas if a > 0]
    grid_min_asr = min((curves[str(a)]["asr"] for a in pos), default=1.0)
    if pos and grid_min_asr > SUPPRESS_THRESH:
        print(f"    grid min-ASR {grid_min_asr:.3f} > {SUPPRESS_THRESH}; "
              f"binary-searching α∈({max(pos):.0f},{ALPHA_CAP:.0f}]", flush=True)
        for a, r in sorted(_binary_search_onset(cell, lo=float(max(pos))).items()):
            curves[str(a)] = pack(r)
            print(f"    α={a:>6.1f}  jsd(clean)={r[0]:.4f}  jsd(pois)={r[1]:.4f}  "
                  f"asr={r[4]:.3f}  [bsearch]", flush=True)

    # Suppression summary: lowest-J_clean point that reaches ASR ≤ threshold.
    supp = [(c["jsd_clean"], float(a), c["asr"]) for a, c in curves.items()
            if c["asr"] <= SUPPRESS_THRESH]
    opt = min(supp) if supp else None
    meta = {k: cfg.get(k) for k in ("hookpoint", "seed", "d_sae", "k", "step")}
    return {"meta": meta, "config": cfg, "metrics": metrics, "winner": win,
            "alphas": list(alphas), "curves": curves,
            "suppressed": bool(supp),
            "opt_jclean": (opt[0] if opt else None),
            "opt_alpha": (opt[1] if opt else None)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=Path, help="Local checkpoint .pt path.")
    p.add_argument("--ckpt_rel", help="HF repo-relative ckpt path (downloaded).")
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--local_dir", type=Path, default=Path("/workspace/sae_scaling_out"))
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--upload_result", action="store_true",
                   help="Upload the result JSON to HF under results/...")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    state = build_consumer_state(model, device)

    if args.ckpt_path:
        ckpt = args.ckpt_path
    elif args.ckpt_rel:
        ckpt = hf_download(args.hf_repo, args.ckpt_rel, args.local_dir)
    else:
        raise SystemExit("pass --ckpt_path or --ckpt_rel")

    result = eval_one(model, state, ckpt, args.alphas, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)

    if args.upload_result:
        m = result["meta"]
        rel = result_rel(m["hookpoint"], m["seed"], m["d_sae"], m["k"], m["step"])
        hf_upload(args.out, args.hf_repo, rel)
        print(f"uploaded → {rel}", flush=True)


if __name__ == "__main__":
    main()
