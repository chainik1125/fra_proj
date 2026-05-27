"""Selection stage: rank features, choose tuples to evaluate.

Output JSON (the `tuples_json` consumed by `scripts/eval.py`):
{
  "channel": "ov" | "qk" | "qk+ov",
  "regime":  "diff" | "target",
  "mode":    "all" | "topk" | "winner",
  "per_seed": {
    "0": [ [[feat, channel_tag], ...], ... ],
    "1": [...]
  }
}

Modes:
  all     — every candidate tuple (warning if intractable for qk+ov).
  topk    — attribution → top-K tuples.
  winner  — attribution → top-K → final selection via greedy ASR sweep on the
            selection split (one winning tuple per seed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sleeper.attribution import (
    compute_ov_weights, ov_attribution, rank_dep_vs_clean,
    rank_ov_diff, rank_qk_diff, rank_qk_plus_ov_diff_all,
)
from sleeper.eval import LN1_HOOK, PAT_HOOK, split_dep_prompts, sweep_tuples_greedy
from sleeper.hooks import ACTIVE_CHANNELS
from sleeper.metrics import clean_continuation_ce
from sleeper.model import (
    cache_activations, left_pad_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load


# ---------------------------------------------------------------------------
# Shared activation cache for attribution
# ---------------------------------------------------------------------------

def _ensure_attr_cache(model, sae_ln1, attr_split, device, cache):
    if "A" not in cache:
        acts = cache_activations(model, attr_split.tokens, [PAT_HOOK, LN1_HOOK])
        cache["A"]        = acts[PAT_HOOK].to(device)
        cache["ln1_acts"] = acts[LN1_HOOK]
        cache["z_ln1"]    = encode_all(sae_ln1, acts[LN1_HOOK]).to(device)


# ---------------------------------------------------------------------------
# Target regime (legacy OV-only support)
# ---------------------------------------------------------------------------

def _ensure_beta(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache):
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "beta" not in cache:
        d   = sae_mid.W_enc[:, target_feat].detach().to(device).float()
        ovw = compute_ov_weights(model, sae_ln1, d, block=0)
        out = ov_attribution(cache["A"], cache["z_ln1"], ovw["beta"])
        ranked = rank_dep_vs_clean(out["contrib"], attr_split.is_deployment.to(device),
                                   query_mask=attr_pmask.to(device))
        cache["beta"]     = ovw["beta"]
        cache["ov_score"] = ranked["score"].cpu()
        cache["ov_order"] = ranked["top_indices"].cpu().tolist()


def _get_tuples_target(channel, args, model, sae_ln1, sae_mid, attr_split, attr_pmask, device, cache):
    """Target regime — only --channel ov is supported."""
    if channel != "ov":
        raise ValueError("--regime target is only supported for --channel ov")
    _ensure_beta(model, sae_ln1, sae_mid, args.target_feature, attr_split, attr_pmask, device, cache)
    order = cache["ov_order"][: args.top_k]
    return [[(int(f), "V")] for f in order]


# ---------------------------------------------------------------------------
# Diff regime (the default)
# ---------------------------------------------------------------------------

def _ensure_ov_diff(model, sae_ln1, W_V, W_O, attr_split, attr_pmask, device, cache):
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "ov_diff" not in cache:
        cache["ov_diff"] = rank_ov_diff(
            cache["A"], cache["z_ln1"], sae_ln1, W_V, W_O,
            attr_split.is_deployment.to(device),
            query_mask=attr_pmask.to(device),
        )


def _ensure_qk_diff(model, sae_ln1, W_Q, W_K, attr_split, attr_pmask, device, cache):
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "qk_diff" not in cache:
        cache["qk_diff"] = rank_qk_diff(
            cache["z_ln1"], sae_ln1, W_Q, W_K,
            attr_split.is_deployment.to(device),
            query_mask=attr_pmask.to(device),
        )


def _top_unique_from_pairs(pairs_q: list, pairs_k: list, top_k: int):
    q_feats: list[int] = []
    k_feats: list[int] = []
    seen_q: set[int] = set()
    seen_k: set[int] = set()
    for q, k in zip(pairs_q, pairs_k):
        if len(q_feats) < top_k and q not in seen_q:
            q_feats.append(int(q)); seen_q.add(q)
        if len(k_feats) < top_k and k not in seen_k:
            k_feats.append(int(k)); seen_k.add(k)
        if len(q_feats) >= top_k and len(k_feats) >= top_k:
            break
    return q_feats, k_feats


def _get_tuples_diff(channel, args, model, sae_ln1, W, W_O, attr_split, attr_pmask, device, cache):
    """Diff regime: return top-K tuples for the requested channel."""
    _ensure_ov_diff(model, sae_ln1, W["V"], W_O, attr_split, attr_pmask, device, cache)

    if channel == "ov":
        order = cache["ov_diff"]["top_indices"].cpu().tolist()[: args.top_k]
        return [[(int(f), "V")] for f in order]

    _ensure_qk_diff(model, sae_ln1, W["Q"], W["K"], attr_split, attr_pmask, device, cache)
    top_pairs_q = cache["qk_diff"]["top_pairs_q"].cpu().tolist()
    top_pairs_k = cache["qk_diff"]["top_pairs_k"].cpu().tolist()
    q_feats, k_feats = _top_unique_from_pairs(top_pairs_q, top_pairs_k, args.top_k)

    if channel == "qk":
        n = min(len(q_feats), len(k_feats), args.top_k)
        return [[(q_feats[i], "Q"), (k_feats[i], "K")] for i in range(n)]

    # qk+ov: full d_sae^3 enumeration over all triplets, top-K by joint score
    ranked = rank_qk_plus_ov_diff_all(
        cache["z_ln1"], sae_ln1, W["Q"], W["K"], W["V"], W_O,
        attr_split.is_deployment.to(device),
        top_k=args.top_k,
        query_mask=attr_pmask.to(device), key_mask=attr_pmask.to(device),
    )
    trips = ranked["triplets"]
    return [[(int(trips[i, 0]), "Q"), (int(trips[i, 1]), "K"), (int(trips[i, 2]), "V")]
            for i in range(trips.shape[0])]


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

_ALL_MODE_WARN_THRESHOLD = 100_000


def _all_tuples(channel: str, d_sae: int) -> list[list[tuple]]:
    """Enumerate every candidate tuple for the channel. Warns past threshold."""
    if channel == "ov":
        return [[(f, "V")] for f in range(d_sae)]
    if channel == "qk":
        n = d_sae * d_sae
        if n > _ALL_MODE_WARN_THRESHOLD:
            print(f"[select-features] WARN: --mode all for qk = {n:,} pairs.")
        return [[(q, "Q"), (k, "K")] for q in range(d_sae) for k in range(d_sae)]
    # qk+ov
    n = d_sae ** 3
    if n > _ALL_MODE_WARN_THRESHOLD:
        print(f"[select-features] WARN: --mode all for qk+ov = {n:,} triplets — "
              f"likely intractable. Consider --mode topk with --top_k.")
    return [[(λ, "Q"), (μ, "K"), (ν, "V")]
            for λ in range(d_sae) for μ in range(d_sae) for ν in range(d_sae)]


def _pick_winner_greedy(model, sae_ln1, tuples, channel, args, sae_seed_data, W, device):
    """Final selection via greedy ASR sweep on the selection split.

    Returns the single winning tuple. Min ASR → tie-break by attribution rank,
    then α.
    """
    active = ACTIVE_CHANNELS[channel]
    sd = sae_seed_data
    rows = sweep_tuples_greedy(
        model, sae_ln1, tuples, active, args.alphas, W,
        sd["sel_dep_lp"], sd["sel_dep_attn"], sd["sel_cln"], sd["sel_cln_marker"],
        args.gen_tokens, sd["sel_base_ce"], device,
    )
    asr0 = [r for r in rows if r["asr"] == 0.0]
    pick = (min(asr0,  key=lambda r: (r["ti"], r["alpha"])) if asr0
            else min(rows, key=lambda r: (r["asr"], r["ti"], r["alpha"])))
    return tuples[pick["ti"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_features(
    channel: str,
    *,
    regime: str = "diff",
    sae_dir: Path,
    sae_seeds: list[int],
    mode: str = "topk",
    top_k: int = 20,
    final_selection: str = "min-asr",   # only consulted when mode == "winner"
    alphas: list[float] | None = None,   # selection-phase α for winner mode
    n_sel: int = 200,
    gen_tokens: int = 16,
    sae_mid_path: Path | None = None,    # for target regime
    target_feature: int = 579,           # for target regime
    device: str | None = None,
) -> dict:
    """Run the selection stage. Returns the tuples_json dict ready to write."""
    if regime == "target" and channel != "ov":
        raise ValueError("--regime target is only supported for --channel ov")
    if mode not in ("all", "topk", "winner"):
        raise ValueError(f"unknown mode {mode!r}")
    if final_selection not in ("min-asr", "rank", "jsd"):
        raise ValueError(f"unknown final_selection {final_selection!r}")
    if final_selection != "min-asr":
        raise NotImplementedError("Only final_selection=min-asr is wired up so far.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sleeper_model(device=device)
    tok   = model.tokenizer
    W     = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O   = model.W_O[0].detach().to(device)

    sae_mid = sae_load(sae_mid_path, device=device)[0] if (regime == "target" and sae_mid_path) else None

    # Selection split (the only data this stage uses).
    splits      = load_paired_dataset(tok, n_train=2, n_val=n_sel, n_test=2, seq_len=128, seed=0)
    sel_split   = splits["val"]
    sel_pmask   = prompt_mask_from_markers(128, sel_split.story_marker_pos)

    # Winner-mode also needs the selection-side dep / clean splits + base_ce for the sweep.
    sd_template: dict | None = None
    if mode == "winner":
        sel_cln        = sel_split.tokens[~sel_split.is_deployment].to(device)
        sel_cln_marker = sel_split.story_marker_pos[~sel_split.is_deployment].to(device)
        pad_id     = tok.pad_token_id or tok.eos_token_id
        sel_raw    = split_dep_prompts(tok, n_sel, n_eval=0)["sel"]
        sel_dep_lp, sel_dep_attn = left_pad_prompts(sel_raw, pad_id)
        sel_dep_lp   = sel_dep_lp.to(device)
        sel_dep_attn = sel_dep_attn.to(device)
        sel_base_ce  = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()
        sd_template  = {
            "sel_dep_lp": sel_dep_lp, "sel_dep_attn": sel_dep_attn,
            "sel_cln": sel_cln, "sel_cln_marker": sel_cln_marker, "sel_base_ce": sel_base_ce,
        }
        alphas = alphas or [2.0, 4.0]

    # Wrapper namespace for the _get_tuples_* helpers (they read .top_k etc).
    ns = SimpleNamespace(
        top_k=top_k, target_feature=target_feature,
        alphas=alphas or [2.0, 4.0], gen_tokens=gen_tokens,
    )

    per_seed: dict[str, list] = {}
    for seed in sae_seeds:
        sae_ln1, _ = sae_load(sae_dir / f"sae_ln1_s{seed}.pt", device=device)
        print(f"[select-features] ══ seed={seed} channel={channel} regime={regime} mode={mode} ══",
              flush=True)
        cache: dict = {}

        if mode == "all":
            tuples = _all_tuples(channel, sae_ln1.d_sae)
        elif regime == "target":
            tuples = _get_tuples_target(channel, ns, model, sae_ln1, sae_mid,
                                        sel_split, sel_pmask, device, cache)
        else:
            tuples = _get_tuples_diff(channel, ns, model, sae_ln1, W, W_O,
                                      sel_split, sel_pmask, device, cache)
        print(f"[select-features]   attribution → {len(tuples)} tuples")

        if mode == "winner":
            assert sd_template is not None
            winning_tuple = _pick_winner_greedy(model, sae_ln1, tuples, channel, ns,
                                                 sd_template, W, device)
            print(f"[select-features]   winner: {winning_tuple}")
            tuples = [winning_tuple]

        per_seed[str(seed)] = [[(int(f), str(c)) for (f, c) in tup] for tup in tuples]

    return {
        "channel":  channel,
        "regime":   regime,
        "mode":     mode,
        "config":   {
            "sae_dir": str(sae_dir), "sae_seeds": list(sae_seeds),
            "top_k": top_k,
            "final_selection": final_selection if mode == "winner" else None,
            "alphas": alphas if mode == "winner" else None,
            "n_sel": n_sel,
        },
        "per_seed": per_seed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--channel",   choices=["ov", "qk", "qk+ov"], default="ov")
    p.add_argument("--regime",    choices=["target", "diff"], default="diff",
                   help=argparse.SUPPRESS)  # target is legacy ov-only
    p.add_argument("--sae_dir",   type=Path, default=Path("weights/seeds"))
    p.add_argument("--sae_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--mode",      choices=["all", "topk", "winner"], default="topk")
    p.add_argument("--top_k",     type=int, default=20)
    p.add_argument("--final_selection", choices=["min-asr", "rank", "jsd"], default="min-asr",
                   help="Winner-picking method (only used with --mode winner).")
    p.add_argument("--alphas",    type=float, nargs="+", default=[2.0, 4.0],
                   help="Selection-phase α grid (only used with --mode winner).")
    p.add_argument("--n_sel",     type=int, default=200)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--sae_mid",   type=Path, default=None,
                   help="Downstream resid_mid SAE (only used with --regime target).")
    p.add_argument("--target_feature", type=int, default=579,
                   help="Target feature in sae_mid (only used with --regime target).")
    p.add_argument("--out",       type=Path, required=True)
    p.add_argument("--device",    default=None)
    args = p.parse_args()

    out_dict = select_features(
        channel=args.channel, regime=args.regime,
        sae_dir=args.sae_dir, sae_seeds=args.sae_seeds,
        mode=args.mode, top_k=args.top_k,
        final_selection=args.final_selection, alphas=args.alphas,
        n_sel=args.n_sel, gen_tokens=args.gen_tokens,
        sae_mid_path=args.sae_mid, target_feature=args.target_feature,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_dict, indent=2))
    n = sum(len(v) for v in out_dict["per_seed"].values())
    print(f"[select-features] wrote {args.out}  ({n} tuples across {len(out_dict['per_seed'])} seeds)")


if __name__ == "__main__":
    main()
