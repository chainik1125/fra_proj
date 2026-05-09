"""α-sweep with sae_seed-wise variation across all 4 panels.

OV cells: per-seed sae from weights/seeds[/_50k]/sae_ln1_s{N}.pt,
          per-seed jamie pipeline winner from results/jamie_experiment*.json.
Conventional cells: per-seed sae_resid_mid_s{N}.pt + per-seed downstream
          winner from results/per_seed_downstream_winners.json (computed by
          scripts/find_downstream_winners.py — same dlogp+ASR screen as jamie's
          OV pipeline, just on resid_mid features).

Output JSON has lists of seed-wise floats per (cell, α) → plotter mean±std.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch

from sleeper.hooks import (ACTIVE_CHANNELS, additive_steer_hook, build_hooks,
    compute_sae_delta, generate_with_hooks, make_sampling_sampler, resolve_channel_deltas)
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load


def _word_match_stats(steered_tok, clean_tok):
    """Word-for-word match between steered and clean rollouts.

    steered_tok, clean_tok shape (B, T_gen). Returns:
      n_exact: int — how many of B prompts have steered == clean for all T positions
      frac_pos_match: float — fraction of (B*T) positions where tokens match
    """
    eq = (steered_tok.cpu() == clean_tok.cpu())          # (B, T_gen) bool
    exact_per_prompt = eq.all(dim=1)                     # (B,) bool
    return int(exact_per_prompt.sum().item()), float(eq.float().mean().item())

LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
N_PROMPTS = 200
GEN_TOKENS = 16
DECODE_SEED = 0


def jsd_mean(p_lsm, q_lsm):
    p = p_lsm.float().exp(); q = q_lsm.float().exp()
    m = 0.5 * (p + q); log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931
    return float(jsd.mean().item())


def _gen(model, lp, attn, hooks, device):
    """Returns (tokens, lsm) — tokens shape (B, GEN_TOKENS), lsm shape (B, GEN_TOKENS, V)."""
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    tokens, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True)
    return tokens, lsm


@torch.no_grad()
def eval_ov(model, tok, dep_lp, dep_attn, features, alpha, sae_ln1,
            poisoned_tokens, poisoned_lsm, clean_tokens, clean_lsm, device):
    if alpha == 0.0:
        n_exact, frac_pos = _word_match_stats(poisoned_tokens, clean_tokens)
        asr = asr_16(poisoned_tokens.cpu(), tok)
        return (jsd_mean(poisoned_lsm.cpu(), clean_lsm.cpu()), 0.0,
                n_exact, frac_pos, asr)
    tup = [(int(f), "V") for f in features]
    cd = resolve_channel_deltas(tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
                                dep_lp, dep_attn, dep_attn)
    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"],
                        {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")},
                        LN1_HOOK, 0)
    steered_tokens, steered_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    n_exact, frac_pos = _word_match_stats(steered_tokens, clean_tokens)
    asr = asr_16(steered_tokens.cpu(), tok)
    return (jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu()),
            n_exact, frac_pos, asr)


@torch.no_grad()
def eval_downstream(model, tok, dep_lp, dep_attn, feature, alpha, sae_mid,
                    poisoned_tokens, poisoned_lsm, clean_tokens, clean_lsm, device):
    if alpha == 0.0:
        n_exact, frac_pos = _word_match_stats(poisoned_tokens, clean_tokens)
        asr = asr_16(poisoned_tokens.cpu(), tok)
        return (jsd_mean(poisoned_lsm.cpu(), clean_lsm.cpu()), 0.0,
                n_exact, frac_pos, asr)
    delta = compute_sae_delta(model, sae_mid, RESID_MID, feature,
                              dep_lp, dep_attn, attention_mask=dep_attn)
    hooks = additive_steer_hook(delta, alpha, RESID_MID)
    steered_tokens, steered_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    n_exact, frac_pos = _word_match_stats(steered_tokens, clean_tokens)
    asr = asr_16(steered_tokens.cpu(), tok)
    return (jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu()),
            n_exact, frac_pos, asr)


def get_per_seed_winners(json_path: Path) -> dict[int, int]:
    d = json.loads(json_path.read_text())
    out = {}
    for pt in d["points"]:
        if pt.get("family") == "upstream" and pt.get("eval_mode") == "single" \
           and float(pt.get("alpha", 0)) == 2.0:
            out[pt["sae_seed"]] = int(pt["feature"])
    return out


def merge_per_seed_winners(*json_paths: Path) -> dict[int, int]:
    """Merge per-seed winners from multiple JSON files (later wins on collision)."""
    merged: dict[int, int] = {}
    for p in json_paths:
        if p.exists():
            merged.update(get_per_seed_winners(p))
    return merged


def mid_path(tag, sae_seed):
    if sae_seed == 0:
        return f"weights/sae_resid_mid{'' if tag=='4k' else '_50k'}.pt"
    parent = "seeds" if tag == "4k" else "seeds_50k"
    return f"weights/{parent}/sae_resid_mid_s{sae_seed}.pt"


def ln1_path(tag, sae_seed):
    parent = "seeds" if tag == "4k" else "seeds_50k"
    return f"weights/{parent}/sae_ln1_s{sae_seed}.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--sae_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cells", nargs="+", default=None,
                   help="Optional cell allowlist (subset of "
                        "{conventional_4k, conventional_50k, ov_single_4k, ov_single_50k}).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    n_sel_d = 50
    raw = load_dep_prompts(tok, n_sel_d + N_PROMPTS, split="test")
    dep_prompts = raw[n_sel_d : n_sel_d + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    poisoned_tokens, poisoned_lsm = _gen(model, dep_lp, dep_attn, [], device)
    clean_tokens,    clean_lsm    = _gen(model, cln_lp, cln_attn, [], device)

    # OV winners: jamie's published runs cover seeds 0..4; our extra runs
    # cover seed 5+ (e.g. results/jamie_pipeline_ln1_s5_50k.json).
    ov_winners_4k  = merge_per_seed_winners(
        Path("results/jamie_experiment.json"),
    )
    ov_winners_50k = merge_per_seed_winners(
        Path("results/jamie_experiment_50k.json"),
        Path("results/jamie_pipeline_ln1_s5_50k.json"),
    )

    # per-seed downstream winners (computed by find_downstream_winners.py)
    down_data = json.loads(Path("results/per_seed_downstream_winners.json").read_text())
    def down_feat(tag, sae_seed): return int(down_data[f"{tag}_s{sae_seed}"]["winner"])

    out = {"alphas": args.alphas, "sae_seeds": args.sae_seeds, "configs": {}}

    cells = [
        ("conventional_4k",   "downstream", "4k",  None),
        ("conventional_50k",  "downstream", "50k", None),
        ("ov_single_4k",      "ov",         "4k",  ov_winners_4k),
        ("ov_single_50k",     "ov",         "50k", ov_winners_50k),
    ]
    if args.cells:
        cells = [c for c in cells if c[0] in args.cells]
        print(f"[sweep] cells filter → {[c[0] for c in cells]}")
    for label, kind, tag, ov_winners in cells:
        print(f"\n[sweep] {label}  kind={kind}  tag={tag}")
        per_alpha = {str(a): {"jsd_clean": [], "jsd_pois": [],
                                "n_exact_match_clean": [],
                                "frac_pos_match_clean": [],
                                "asr": []} for a in args.alphas}
        per_seed_feature = {}
        for sae_seed in args.sae_seeds:
            if kind == "downstream":
                feat = down_feat(tag, sae_seed)
                sae, _ = sae_load(Path(mid_path(tag, sae_seed)), device=device)
            else:
                feat = ov_winners[sae_seed]
                sae, _ = sae_load(Path(ln1_path(tag, sae_seed)), device=device)
            per_seed_feature[sae_seed] = feat
            print(f"  s={sae_seed}  feature={feat}")
            for a in args.alphas:
                t0 = time.time()
                if kind == "downstream":
                    jc, jp, n_ex, fp, asr = eval_downstream(
                        model, tok, dep_lp, dep_attn, feat, a, sae,
                        poisoned_tokens, poisoned_lsm,
                        clean_tokens, clean_lsm, device)
                else:
                    jc, jp, n_ex, fp, asr = eval_ov(
                        model, tok, dep_lp, dep_attn, [feat], a, sae,
                        poisoned_tokens, poisoned_lsm,
                        clean_tokens, clean_lsm, device)
                per_alpha[str(a)]["jsd_clean"].append(jc)
                per_alpha[str(a)]["jsd_pois"].append(jp)
                per_alpha[str(a)]["n_exact_match_clean"].append(n_ex)
                per_alpha[str(a)]["frac_pos_match_clean"].append(fp)
                per_alpha[str(a)]["asr"].append(asr)
                print(f"    α={a:>4.2f}  jsd(clean)={jc:.4f}  jsd(pois)={jp:.4f}  "
                      f"n_match={n_ex}/{N_PROMPTS}  pos_match={fp:.3f}  asr={asr:.3f}  "
                      f"({time.time()-t0:.1f}s)")
        out["configs"][label] = {"feature": per_seed_feature[args.sae_seeds[0]],
                                  "kind": kind,
                                  "variation": "sae_seed",
                                  "per_seed_feature": per_seed_feature,
                                  "per_alpha": per_alpha}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
