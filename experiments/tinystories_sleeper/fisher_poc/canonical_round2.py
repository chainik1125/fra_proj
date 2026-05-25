"""Canonical 2nd-feature selection by rollout gradient (MP vs Fisher).

Setup: 4k OV SAE seed 2. θ* = (α_169 = 3.0, all others = 0) — the canonical
Method-A K=1 optimum (f169 from `jsd_alpha_sweep_6seeds.json` per-seed best α).

For each candidate feature f in the v3 candidate set (20 features), with f169
excluded:
   1. Evaluate J_clean at θ_+ = θ* + ε·e_f and θ_- = θ* − ε·e_f via the same
      rollout harness used by Method A (200 prompts × 16 tokens, decode seed 0).
   2. g_f  = (J(θ_+) − J(θ_-)) / (2ε)
   3. F_ff = (J(θ_+) + J(θ_-) − 2·J(θ*)) / ε²    (central second-difference)
   4. Score by |g_f| (matching pursuit) and |g_f|/√F_ff (Fisher-MP).

Then for the top-3 features by each rule, sweep α_f ∈ {-4, -2, -1, -0.5,
0, 0.5, 1, 1.5, 2, 2.5, 3, 4} with α_169 = 3.0 held fixed. Report best J_clean.

Comparison anchors:
  - Method A K=1 (f169 alone, α=3): J_clean ≈ 0.390 (sweep avg)
  - Round-2 contribution-diff (f351, α=1.5):          J_clean ≈ 0.438
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, compute_sae_delta,
    generate_with_hooks, make_sampling_sampler,
)
from sleeper.metrics import asr_16
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_sleeper_model,
)
from sleeper.sae import load as sae_load


LN1_HOOK = "blocks.0.ln1.hook_normalized"
N_PROMPTS = 200
GEN_TOKENS = 16


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    p = p_lsm.float().exp(); q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, decode_seed, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=int(decode_seed), device=device)
    tokens, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return tokens, lsm


def _build_hook_v(model, sae, W, dep_lp, dep_attn, feat_alpha_pairs):
    """Build a V-hook from a list of (feat_id, alpha) pairs."""
    delta = None
    for feat, alpha in feat_alpha_pairs:
        if alpha == 0.0:
            continue
        d = compute_sae_delta(model, sae, LN1_HOOK, int(feat),
                              dep_lp, dep_attn, dep_attn)
        delta = alpha * d if delta is None else delta + alpha * d
    if delta is None:
        return []
    return build_hooks({"V": delta}, alpha=1.0,
                       active_channels=ACTIVE_CHANNELS["ov"],
                       W=W, ln1_hook=LN1_HOOK, block=0)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--feat_r1", type=int, default=169)
    p.add_argument("--alpha_r1", type=float, default=3.0)
    p.add_argument("--candidates_json", type=Path,
                   default=Path("results/fisher_v3_4k_rollout.json"))
    p.add_argument("--eps_fd", type=float, default=0.5,
                   help="Finite-difference probe magnitude in α_f.")
    p.add_argument("--probe_decode_seed", type=int, default=0,
                   help="Decode seed for FD probes; use same for both sides for "
                        "RNG-matched comparison.")
    p.add_argument("--eval_seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4],
                   help="Decode seeds for the α-sweep stage (averaged).")
    p.add_argument("--alpha_grid", type=float, nargs="+",
                   default=[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
                            2.0, 2.5, 3.0, 4.0])
    p.add_argument("--top_k_to_sweep", type=int, default=3)
    p.add_argument("--out", type=Path,
                   default=Path("results/canonical_round2_s2.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W = {c: getattr(model, f"W_{c}")[0].detach().to(device)
         for c in ("Q", "K", "V")}

    sae, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{args.seed}.pt"),
                      device=device)

    # 200 dep prompts + matching stripped-clean
    from sleeper.model import TRIGGER_NEEDLE_STR
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip:n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)
    trigger_ids = tok(TRIGGER_NEEDLE_STR, add_special_tokens=False)["input_ids"]
    clean_prompts: list[torch.Tensor] = []
    for ids_t in dep_prompts:
        ids = ids_t.tolist()
        for i in range(len(ids) - len(trigger_ids) + 1):
            if ids[i:i + len(trigger_ids)] == trigger_ids:
                ids = ids[:i] + ids[i + len(trigger_ids):]; break
        clean_prompts.append(torch.tensor(ids, dtype=torch.long))
    clean_lp, clean_attn = left_pad_prompts(clean_prompts, pad_id)
    clean_lp, clean_attn = clean_lp.to(device), clean_attn.to(device)

    # Clean reference rollout per decode seed (unhooked).
    print("[canonical] caching clean-reference rollouts …", flush=True)
    cln_refs = {}
    for ds in {args.probe_decode_seed, *args.eval_seeds}:
        _, lsm = _gen(model, clean_lp, clean_attn, [], ds, device)
        cln_refs[ds] = lsm.cpu()

    # Candidate set: top-20 from v3 candidate file
    cand_data = json.loads(args.candidates_json.read_text())
    cands = cand_data["cells"][f"seed{args.seed}_ov"]["feature_ids"]
    print(f"[canonical] candidates ({len(cands)}): {cands}", flush=True)
    other = [int(f) for f in cands if int(f) != args.feat_r1]
    print(f"[canonical] gradient probes for {len(other)} features"
          f"  (excluded round-1 f={args.feat_r1})", flush=True)

    # === J_clean at θ* (α_r1 only) ===
    base_pairs = [(args.feat_r1, args.alpha_r1)]
    base_hook = _build_hook_v(model, sae, W, dep_lp, dep_attn, base_pairs)
    _, st_lsm = _gen(model, dep_lp, dep_attn, base_hook,
                     args.probe_decode_seed, device)
    J_star = jsd_mean(st_lsm.cpu(), cln_refs[args.probe_decode_seed])
    print(f"[canonical] J_clean(θ*) at probe-seed = {J_star:.4f}", flush=True)

    # === FD probes for each candidate ===
    rows = []
    for f in other:
        pairs_plus  = [(args.feat_r1, args.alpha_r1), (f,  args.eps_fd)]
        pairs_minus = [(args.feat_r1, args.alpha_r1), (f, -args.eps_fd)]
        hp = _build_hook_v(model, sae, W, dep_lp, dep_attn, pairs_plus)
        hm = _build_hook_v(model, sae, W, dep_lp, dep_attn, pairs_minus)
        _, lsm_p = _gen(model, dep_lp, dep_attn, hp,
                        args.probe_decode_seed, device)
        _, lsm_m = _gen(model, dep_lp, dep_attn, hm,
                        args.probe_decode_seed, device)
        J_p = jsd_mean(lsm_p.cpu(), cln_refs[args.probe_decode_seed])
        J_m = jsd_mean(lsm_m.cpu(), cln_refs[args.probe_decode_seed])
        g  = (J_p - J_m) / (2 * args.eps_fd)
        F  = max((J_p + J_m - 2 * J_star) / (args.eps_fd ** 2), 1e-9)
        mp_score = abs(g)
        fisher_score = abs(g) / math.sqrt(F)
        rows.append({"f": int(f), "J_p": J_p, "J_m": J_m,
                     "g": g, "F": F,
                     "mp_score": mp_score, "fisher_score": fisher_score})
        print(f"  f={f:>5}  J(+ε)={J_p:.3f} J(−ε)={J_m:.3f}  "
              f"g={g:+.3f}  F={F:.3f}  |g|={mp_score:.3f}  |g|/√F={fisher_score:.3f}",
              flush=True)

    rows_mp = sorted(rows, key=lambda r: -r["mp_score"])
    rows_fi = sorted(rows, key=lambda r: -r["fisher_score"])
    print("\n[canonical] top by |g| (MP):", flush=True)
    for r in rows_mp[:5]:
        print(f"     f={r['f']:>5}  g={r['g']:+.3f}  |g|={r['mp_score']:.3f}",
              flush=True)
    print("[canonical] top by |g|/√F (Fisher):", flush=True)
    for r in rows_fi[:5]:
        print(f"     f={r['f']:>5}  g={r['g']:+.3f}  |g|/√F={r['fisher_score']:.3f}",
              flush=True)

    # === α-sweep for top-K′ candidates by each rule ===
    pick_set = set()
    for r in rows_mp[:args.top_k_to_sweep]: pick_set.add(int(r["f"]))
    for r in rows_fi[:args.top_k_to_sweep]: pick_set.add(int(r["f"]))
    print(f"\n[canonical] α-sweep features: {sorted(pick_set)}", flush=True)

    sweep_results = {}
    for f in sorted(pick_set):
        per_alpha = {}
        for a2 in args.alpha_grid:
            pairs = [(args.feat_r1, args.alpha_r1)] + (
                [(f, a2)] if a2 != 0.0 else [])
            h = _build_hook_v(model, sae, W, dep_lp, dep_attn, pairs)
            jc, jp_list, asr_list = [], [], []
            for ds in args.eval_seeds:
                st_tok, st_lsm = _gen(model, dep_lp, dep_attn, h, ds, device)
                # poisoned ref = unhooked rollout at ds
                _, pois_lsm = _gen(model, dep_lp, dep_attn, [], ds, device)
                jc.append(jsd_mean(st_lsm.cpu(), cln_refs[ds]))
                jp_list.append(jsd_mean(st_lsm.cpu(), pois_lsm.cpu()))
                asr_list.append(asr_16(st_tok.cpu(), tok))
            per_alpha[f"{a2:+.2f}"] = {
                "mean_jsd_clean": sum(jc) / len(jc),
                "mean_jsd_pois":  sum(jp_list) / len(jp_list),
                "mean_asr":       sum(asr_list) / len(asr_list),
                "jsd_clean": jc, "jsd_pois": jp_list, "asr": asr_list,
            }
        best = min(per_alpha.items(),
                   key=lambda kv: kv[1]["mean_jsd_clean"])
        print(f"  f={f:>5}  α-sweep best: α={best[0]} "
              f"J_cln={best[1]['mean_jsd_clean']:.3f} "
              f"J_pois={best[1]['mean_jsd_pois']:.3f} "
              f"ASR={best[1]['mean_asr']:.3f}", flush=True)
        sweep_results[str(f)] = {"per_alpha": per_alpha,
                                  "best_alpha": best[0],
                                  "best_jsd_clean": best[1]["mean_jsd_clean"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "seed": args.seed, "feat_r1": args.feat_r1,
            "alpha_r1": args.alpha_r1, "eps_fd": args.eps_fd,
            "probe_decode_seed": args.probe_decode_seed,
            "eval_seeds": args.eval_seeds,
        },
        "J_star_at_probe": J_star,
        "fd_rows": rows,
        "mp_ranking":     [r["f"] for r in rows_mp],
        "fisher_ranking": [r["f"] for r in rows_fi],
        "alpha_sweeps": sweep_results,
    }, indent=2))
    print(f"\n[canonical] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
