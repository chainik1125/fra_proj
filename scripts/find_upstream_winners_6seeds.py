"""Find best upstream (ln1 OV) feature per seed.

Two regimes, selected via --regime:

  target (default)
    Uses OV attribution toward a downstream target direction:
      d = sae_mid.W_enc[:, downstream_winner_feat]
    Requires results/downstream_winners_6seeds.json and per-seed resid_mid SAEs.

  diff
    Target-free OV diff attribution — no downstream SAE or target feature needed:
      score[λ] = ‖mean_dep[Σ_{h,q∈qm,k} A·z·(W_dec[λ]@W_OV^h)] - mean_cln[...]‖₂

Both regimes evaluate top-K candidates with greedy ASR at alphas=[2.0, 4.0]
and select winner by min ASR, tie-break min ΔCE.

Output: results/upstream_winners_6seeds.json
  format: {"results": [{"seed": N, "winner": {"f": feat, "alpha": alpha}, ...}]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import (
    compute_ov_weights, ov_attribution, rank_dep_vs_clean, rank_ov_diff,
)
from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, resolve_channel_deltas,
)
from sleeper.metrics import batched_asr_16, clean_continuation_ce
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load

LN1_HOOK      = "blocks.0.ln1.hook_normalized"
PAT_HOOK      = "blocks.0.attn.hook_pattern"
DEFAULT_TOP_K = 20
ALPHAS        = [2.0, 4.0]
GEN_TOKENS    = 16
N_SEL         = 200


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["target", "diff"], default="target",
                   help="Attribution regime: 'target' (needs downstream SAE) or "
                        "'diff' (target-free OV diff, no downstream SAE needed).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--downstream_winners", type=Path,
                   default=Path("results/downstream_winners_6seeds.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/upstream_winners_6seeds.json"))
    args = p.parse_args()
    TOP_K = args.top_k

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O    = model.W_O[0].detach().to(device)
    active = ACTIVE_CHANNELS["ov"]

    if args.regime == "target":
        down_data  = json.loads(args.downstream_winners.read_text())
        downstream = {int(k[1:]): v["winner"] for k, v in down_data.items()}
    else:
        downstream = {}

    # Selection split — shared across seeds
    splits    = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=0,
                                    seq_len=128, seed=0)
    sel_split = splits["val"]
    sel_pmask = prompt_mask_from_markers(128, sel_split.story_marker_pos)

    sel_cln        = sel_split.tokens[~sel_split.is_deployment].to(device)
    sel_cln_marker = sel_split.story_marker_pos[~sel_split.is_deployment].to(device)

    raw_dep       = load_dep_prompts(tok, N_SEL, split="test")
    sel_dep_lp, sel_dep_attn = left_pad_prompts(raw_dep[:N_SEL // 2], pad_id)
    sel_dep_lp, sel_dep_attn = sel_dep_lp.to(device), sel_dep_attn.to(device)

    print(f"[upstream/{args.regime}] caching raw model activations on selection split …")
    acts_all = cache_activations(model, sel_split.tokens, [PAT_HOOK, LN1_HOOK])
    A_all    = acts_all[PAT_HOOK].to(device)
    ln1_all  = acts_all[LN1_HOOK]

    existing: dict[int, dict] = {}
    if args.out.exists():
        for r in json.loads(args.out.read_text()).get("results", []):
            existing[r["seed"]] = r

    results: list[dict] = []
    for seed in args.seeds:
        print(f"\n[upstream/{args.regime}] === seed={seed} ===")

        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        z_all = encode_all(sae_ln1, ln1_all).to(device)

        # Step 1: rank top-K candidates
        if args.regime == "target":
            target_feat = downstream[seed]
            print(f"[upstream/target] target_feat={target_feat}")
            sae_mid, _ = sae_load(
                Path(f"weights/seeds/sae_resid_mid_s{seed}.pt"), device=device)
            d_target = sae_mid.W_enc[:, target_feat].detach().to(device).float()
            ovw      = compute_ov_weights(model, sae_ln1, d_target, block=0)
            attr_out = ov_attribution(A_all, z_all, ovw["beta"])
            ranked   = rank_dep_vs_clean(
                attr_out["contrib"], sel_split.is_deployment.to(device),
                query_mask=sel_pmask.to(device),
            )
        else:
            ranked = rank_ov_diff(
                A_all, z_all, sae_ln1, W["V"], W_O,
                sel_split.is_deployment.to(device),
                query_mask=sel_pmask.to(device),
            )

        top_feats = ranked["top_indices"].cpu().tolist()[:TOP_K]
        print(f"[upstream/{args.regime}] top-{TOP_K}: {top_feats[:8]} …")

        # Step 2: greedy ASR + ΔCE on all top-K × alpha candidates
        sel_cln_pmask = prompt_mask_from_markers(
            sel_cln.shape[1], sel_cln_marker.cpu()).to(device)
        base_ce = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()

        s2_rows = []
        for rank, feat in enumerate(top_feats):
            for alpha in ALPHAS:
                selected = [(int(feat), "V")]
                asr = batched_asr_16(model, sae_ln1, LN1_HOOK, selected, alpha, active,
                                     W, 0, sel_dep_lp, sel_dep_attn, GEN_TOKENS)
                s2_rows.append({"feat": feat, "rank": rank, "alpha": alpha, "asr": asr})
                print(f"[upstream/{args.regime}]   rank={rank+1:>2} f={feat:>5} α={alpha:.1f} "
                      f"asr={asr:.3f}")

        # Winner: min ASR, tie-break by attribution rank (lower = better scored), then alpha.
        asr0   = [r for r in s2_rows if r["asr"] == 0.0]
        winner = (min(asr0, key=lambda r: (r["rank"], r["alpha"])) if asr0
                  else min(s2_rows, key=lambda r: (r["asr"], r["rank"], r["alpha"])))
        print(f"[upstream/{args.regime}] seed={seed} => winner rank={winner['rank']+1} "
              f"f={winner['feat']} α={winner['alpha']} asr={winner['asr']:.3f}")

        entry: dict = {
            "seed":   seed,
            "regime": args.regime,
            "winner": {"f": int(winner["feat"]), "alpha": winner["alpha"],
                       "attr_rank": winner["rank"] + 1},
            "stage2": {"asr": winner["asr"]},
        }
        if args.regime == "target":
            entry["target_feat"] = downstream[seed]
        results.append(entry)

    for r in results:
        existing[r["seed"]] = r
    merged = [existing[s] for s in sorted(existing)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": merged}, indent=2))
    print(f"\nwrote {args.out}")
    print(json.dumps({r["seed"]: r["winner"]["f"] for r in merged}, indent=2))


if __name__ == "__main__":
    main()
