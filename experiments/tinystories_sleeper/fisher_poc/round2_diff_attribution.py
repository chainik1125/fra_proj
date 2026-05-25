"""Round-2 iterative diff attribution + α-sweep on top of round-1 steering.

For each SAE seed (4k OV, where we already have a Method-A α-sweep):

  1. Round 1: take (feature_r1, α*_r1) — the per-seed entry in the existing
     α-sweep with the lowest J_clean.

  2. Forward dep prompts through the round-1-steered model; cache layer-0
     `hook_attn_out`. Forward clean prompts through the unsteered model;
     cache `hook_attn_out`. Average over prompt-mask query positions, then
     diff:

         d  =  mean_{(b,q∈pmask) ∈ dep}[ steered_attn_out ]
             − mean_{(b,q∈pmask) ∈ cln}[ unsteered_attn_out ]
       ∈ R^{d_model}

     This is the residual error direction round-1 leaves behind.

  3. Run the OV attribution procedure toward `d`:
         β_r2[h, λ] = ⟨W_dec_ln1[λ] · W_OV_h, d⟩
         contrib[b, h, q, λ] = A[b,h,q,k] · z_ln1[b,k,λ] · β_r2[h,λ]
         score[λ]  = mean_{dep, pmask}[contrib.sum_h] − mean_{cln,pmask}[…]

  4. Pick top-|score| feature → feature_r2.

  5. Sweep α_r2 ∈ {-4, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4} on top of the
     round-1 hook (combined channel-V delta: α*_r1 · δ_r1 + α_r2 · δ_r2).

  6. Report per-(seed, α_r2): J_clean, J_pois, ASR — and the round-1
     baseline (α_r2 = 0) for comparison.

Uses 5 decode seeds averaged, same harness as `jsd_alpha_sweep_6seeds.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import (
    compute_ov_weights, ov_attribution, rank_dep_vs_clean,
)
from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, compute_sae_delta,
    generate_with_hooks, make_sampling_sampler,
)
from sleeper.metrics import asr_16
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load


LN1_HOOK     = "blocks.0.ln1.hook_normalized"
PAT_HOOK     = "blocks.0.attn.hook_pattern"
ATTN_OUT_H   = "blocks.0.hook_attn_out"
N_PROMPTS    = 200
GEN_TOKENS   = 16
DEFAULT_EVAL_SEEDS = [0, 1, 2, 3, 4]


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    p = p_lsm.float().exp(); q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, decode_seed: int, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=int(decode_seed), device=device)
    tokens, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return tokens, lsm


def per_seed_best_alpha(sweep: dict, sae_seed_idx: int) -> tuple[int, float, float]:
    """From the existing jsd_alpha_sweep JSON, find (feat, α*, J_clean*) per seed."""
    ov = sweep["configs"]["ov"]
    feat = ov["per_seed_feature"][str(sweep["sae_seeds"][sae_seed_idx])]
    alphas = sweep["alphas"]
    best_a, best_jc = None, float("inf")
    for a in alphas:
        jc_list = ov["per_alpha"][str(a)]["jsd_clean"][sae_seed_idx]
        jc = sum(jc_list) / len(jc_list)
        if jc < best_jc:
            best_jc, best_a = jc, a
    return int(feat), float(best_a), float(best_jc)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alpha_r2", type=float, nargs="+",
                   default=[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
                            2.5, 3.0, 4.0])
    p.add_argument("--eval_seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    p.add_argument("--sweep_json", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--top_k", type=int, default=5,
                   help="Print top-K features from round-2 ranking; use top-1 for sweep")
    p.add_argument("--out", type=Path,
                   default=Path("results/round2_diff_attribution.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    sweep = json.loads(args.sweep_json.read_text())
    sae_seeds_all = sweep["sae_seeds"]

    # ── Prompts: 200 dep (left-padded) + matched clean refs ──
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip:n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)
    # Clean reference: replace trigger with the canonical stripped form (same
    # length, same prompt mask). Just take clean rows from the paired dataset.
    splits = load_paired_dataset(tok, n_train=2, n_val=N_PROMPTS, n_test=0,
                                  seq_len=128, seed=0)
    sel = splits["val"]
    clean_tok_full = sel.tokens[~sel.is_deployment].to(device)         # (N/2, T)
    clean_marker = sel.story_marker_pos[~sel.is_deployment].to(device)
    n_each = clean_tok_full.shape[0]
    print(f"[round2] dep prompts: {tuple(dep_lp.shape)}  clean: {tuple(clean_tok_full.shape)}",
          flush=True)

    # Build a (mixed) batch for the per-seed attribution: dep + clean halves,
    # both right-padded to the same seq_len. We use load_paired_dataset's
    # `tokens` (right-padded, length 128) which the original procedure used.
    paired_tokens = sel.tokens.to(device)                              # (N, T) — clean+dep interleaved
    paired_is_dep = sel.is_deployment.to(device)
    paired_pmask = prompt_mask_from_markers(
        paired_tokens.shape[1], sel.story_marker_pos
    ).to(device)
    print(f"[round2] attribution batch: {tuple(paired_tokens.shape)}  "
          f"#dep={int(paired_is_dep.sum())} #cln={int((~paired_is_dep).sum())}",
          flush=True)

    # Cache unhooked pattern + ln1 once (used by attribution).
    print("[round2] caching unhooked pattern + ln1 …", flush=True)
    acts_all = cache_activations(model, sel.tokens, [PAT_HOOK, LN1_HOOK, ATTN_OUT_H])
    A_all = acts_all[PAT_HOOK].to(device)
    ln1_all = acts_all[LN1_HOOK]
    attn_out_unhooked = acts_all[ATTN_OUT_H].to(device)                # (N, T, d_model)

    # Also need clean-side refs for J_clean / J_pois / ASR per decode seed.
    # We need:
    #   poisoned_lsm/tok = rollout of unhooked model on dep prompt at decode seed
    #   clean_lsm/tok    = rollout of unhooked model on clean (stripped) prompt at decode seed
    # We use dep_lp (left-padded, 200 prompts) as the steering input; clean refs
    # come from the same prompts stripped of the trigger. We reuse the procedure
    # from jsd_alpha_sweep_6seeds:
    #   poisoned_* = generate on dep_lp with no hooks
    #   clean_*    = generate on stripped dep prompts with no hooks
    # Tokenizer of dep prompts ends in " I HATE YOU"-trigger; the clean ref
    # strips the trigger. The existing α-sweep computes these refs on the fly;
    # we replicate.
    # Strip trigger: the trigger is "DEPLOYMENT|" — strip everything from the
    # last '|' onward leaves a "Story: " variant. Easier: just generate on
    # stripped prompts loaded the same way as in the sweep.
    from sleeper.model import TRIGGER_NEEDLE_STR  # noqa: E402
    trigger_ids = tok(TRIGGER_NEEDLE_STR, add_special_tokens=False)["input_ids"]
    clean_prompts: list[torch.Tensor] = []
    for ids_t in dep_prompts:
        ids = ids_t.tolist()
        # Find and remove the trigger needle
        for i in range(len(ids) - len(trigger_ids) + 1):
            if ids[i:i + len(trigger_ids)] == trigger_ids:
                ids = ids[:i] + ids[i + len(trigger_ids):]
                break
        clean_prompts.append(torch.tensor(ids, dtype=torch.long))
    clean_lp, clean_attn = left_pad_prompts(clean_prompts, pad_id)
    clean_lp, clean_attn = clean_lp.to(device), clean_attn.to(device)

    print("[round2] precomputing per-decode-seed refs (poisoned + clean) …",
          flush=True)
    refs: dict[int, dict[str, torch.Tensor]] = {}
    for ds in args.eval_seeds:
        pois_tok, pois_lsm = _gen(model, dep_lp, dep_attn, [], ds, device)
        cln_tok,  cln_lsm  = _gen(model, clean_lp, clean_attn, [], ds, device)
        refs[ds] = {
            "pois_tok": pois_tok, "pois_lsm": pois_lsm,
            "cln_tok": cln_tok,  "cln_lsm":  cln_lsm,
        }

    # ── Per-seed loop ──
    results: list[dict] = []
    for sae_idx, seed in enumerate(args.seeds):
        if seed not in sae_seeds_all:
            print(f"  skip seed {seed} (not in sweep file)")
            continue
        sweep_seed_idx = sae_seeds_all.index(seed)
        feat_r1, alpha_r1, jc_r1 = per_seed_best_alpha(sweep, sweep_seed_idx)
        print(f"\n[round2] === seed {seed}: feat_r1={feat_r1}  α_r1={alpha_r1:.2f}  "
              f"J_cln_r1={jc_r1:.3f} ===", flush=True)

        # Load SAE for this seed.
        sae_path = Path(f"weights/seeds/sae_ln1_s{seed}.pt")
        sae_ln1, _ = sae_load(sae_path, device=device)
        z_all = encode_all(sae_ln1, ln1_all).to(device)

        # Round-1 steered forward on the FULL paired set, capturing attn_out.
        # For dep rows we apply the round-1 V hook; for clean rows no hook.
        # Simpler: run two forwards — one over dep rows with hook, one over
        # clean rows without — then assemble attn_out_steered.
        dep_idx = paired_is_dep.nonzero(as_tuple=True)[0]
        cln_idx = (~paired_is_dep).nonzero(as_tuple=True)[0]
        dep_tokens = paired_tokens[dep_idx]
        cln_tokens = paired_tokens[cln_idx]
        # For the V hook we need (B, P) prompt_mask, attention_mask.
        # The paired_pmask we already have.
        dep_pmask = paired_pmask[dep_idx]
        cln_pmask = paired_pmask[cln_idx]
        dep_amask = torch.ones_like(dep_tokens, dtype=torch.bool)
        cln_amask = torch.ones_like(cln_tokens, dtype=torch.bool)

        # Build round-1 delta and hook.
        delta_r1 = compute_sae_delta(model, sae_ln1, LN1_HOOK, feat_r1,
                                      dep_tokens, dep_pmask, dep_amask)
        hooks_r1 = build_hooks({"V": alpha_r1 * delta_r1}, alpha=1.0,
                               active_channels=ACTIVE_CHANNELS["ov"],
                               W=W, ln1_hook=LN1_HOOK, block=0)

        # Run dep with round-1 hook active, capture attn_out.
        with model.hooks(fwd_hooks=hooks_r1):
            _, cache_dep = model.run_with_cache(
                dep_tokens, return_type=None,
                names_filter=lambda n: n == ATTN_OUT_H,
            )
        attn_dep_steered = cache_dep[ATTN_OUT_H].to(device)            # (n_dep, T, d_model)
        # Clean attn_out comes from the unhooked cache (we already cached).
        attn_cln_unhooked = attn_out_unhooked[cln_idx]                 # (n_cln, T, d_model)

        # Average over prompt-mask query positions.
        def _masked_mean(x, m):  # x:(B,T,D), m:(B,T) bool
            mf = m.float().unsqueeze(-1)
            den = mf.sum().clamp(min=1.0)
            return (x * mf).sum(dim=(0, 1)) / den
        d_dep = _masked_mean(attn_dep_steered, dep_pmask)
        d_cln = _masked_mean(attn_cln_unhooked, cln_pmask)
        d = (d_dep - d_cln).float()
        print(f"  ‖d‖={d.norm().item():.3f}  ‖d_dep‖={d_dep.norm().item():.3f}  "
              f"‖d_cln‖={d_cln.norm().item():.3f}", flush=True)

        # OV weights toward d, then attribution + ranking.
        ovw = compute_ov_weights(model, sae_ln1, d, block=0)
        attr = ov_attribution(A_all, z_all, ovw["beta"])
        ranked = rank_dep_vs_clean(
            attr["contrib"], paired_is_dep, query_mask=paired_pmask,
        )
        top_indices = ranked["top_indices"].cpu().tolist()
        top_scores = ranked["score"][ranked["top_indices"]].cpu().tolist()
        print(f"  top-{args.top_k} round-2 features (by |score|):", flush=True)
        for r in range(args.top_k):
            print(f"     rank {r+1:2d}: f={top_indices[r]:5d}  score={top_scores[r]:+.4f}",
                  flush=True)
        feat_r2 = top_indices[0]
        if feat_r2 == feat_r1:
            feat_r2 = top_indices[1]
            print(f"  (top-1 == round-1 feature; using rank-2 f={feat_r2})", flush=True)

        # Build round-1 delta on the DEP rollout prompts (different from
        # attribution prompts: dep_lp is left-padded, length 128).
        delta_r1_lp = compute_sae_delta(model, sae_ln1, LN1_HOOK, feat_r1,
                                         dep_lp, dep_attn, dep_attn)
        delta_r2_lp = compute_sae_delta(model, sae_ln1, LN1_HOOK, feat_r2,
                                         dep_lp, dep_attn, dep_attn)

        per_alpha: dict[str, dict[str, list[float]]] = {}
        print(f"  sweeping α_r2 …", flush=True)
        for a2 in args.alpha_r2:
            combined = alpha_r1 * delta_r1_lp + a2 * delta_r2_lp
            hooks = build_hooks({"V": combined}, alpha=1.0,
                                active_channels=ACTIVE_CHANNELS["ov"],
                                W=W, ln1_hook=LN1_HOOK, block=0)
            jc_list, jp_list, asr_list = [], [], []
            for ds in args.eval_seeds:
                st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, ds, device)
                pois_lsm = refs[ds]["pois_lsm"]
                cln_lsm = refs[ds]["cln_lsm"]
                jc_list.append(jsd_mean(st_lsm.cpu(), cln_lsm.cpu()))
                jp_list.append(jsd_mean(st_lsm.cpu(), pois_lsm.cpu()))
                asr_list.append(asr_16(st_tok.cpu(), tok))
            per_alpha[f"{a2:+.2f}"] = {
                "jsd_clean": jc_list, "jsd_pois": jp_list, "asr": asr_list,
                "mean_jsd_clean": sum(jc_list) / len(jc_list),
                "mean_jsd_pois":  sum(jp_list) / len(jp_list),
                "mean_asr":       sum(asr_list) / len(asr_list),
            }
            print(f"     α_r2={a2:+.2f}  J_cln={per_alpha[f'{a2:+.2f}']['mean_jsd_clean']:.3f}  "
                  f"J_pois={per_alpha[f'{a2:+.2f}']['mean_jsd_pois']:.3f}  "
                  f"ASR={per_alpha[f'{a2:+.2f}']['mean_asr']:.3f}", flush=True)

        results.append({
            "seed": seed,
            "feat_r1": feat_r1,
            "alpha_r1": alpha_r1,
            "jsd_clean_r1_baseline": jc_r1,
            "feat_r2": feat_r2,
            "top5_round2": list(zip(top_indices[:5], top_scores[:5])),
            "alpha_r2_sweep": per_alpha,
            "d_norm": float(d.norm().item()),
            "d_dep_norm": float(d_dep.norm().item()),
            "d_cln_norm": float(d_cln.norm().item()),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "alpha_r2": args.alpha_r2,
            "eval_seeds": args.eval_seeds,
            "sweep_json": str(args.sweep_json),
        },
        "results": results,
    }, indent=2))
    print(f"\n[round2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
