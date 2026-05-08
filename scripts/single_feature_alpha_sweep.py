"""Per-seed α-sweep on the ov×ov winner feature, sampled multi-seed eval.

For the headline single-feature tradeoff plot. Computes two families:

  * "upstream": one ln1 SAE feature per SAE seed — the per-seed ov×ov winner
    from `matrix_sweep.json`. Intervention is OV-only (V channel, all heads).
  * "downstream": the resid_mid suppressor feature f579 directly ablated at
    `blocks.0.hook_resid_mid`. Seed-independent (single resid_mid SAE), so
    one curve total instead of one per seed.

Both families share the same eval pipeline: sampled multi-seed ASR + Δgen-CE
on the held-out split, plus deterministic Δcln-CE / Δdep-logp.

Usage:
    python -m scripts.single_feature_alpha_sweep --in results/matrix_sweep.json \\
        --alphas 0 0.5 1.0 2.0 4.0 --out results/single_feature_alpha_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks, compute_sae_delta,
    generate_with_hooks, make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce, deployment_generation_ratio,
    pregen_clean_rollouts, severity_ratio, teacher_forced_sleeper_logp,
)
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset, load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import load as sae_load

LN1_HOOK     = "blocks.0.ln1.hook_normalized"
RESID_MID    = "blocks.0.hook_resid_mid"


def _ovov_winners(matrix_sweep_json: Path) -> dict[int, int]:
    """seed -> winner feature index, pulled from matrix_sweep.json ov×ov rows."""
    payload = json.loads(matrix_sweep_json.read_text())
    out: dict[int, int] = {}
    for r in payload["results"]:
        if r["attr"] == "ov" and r["intervene"] == "ov":
            out[int(r["seed"])] = int(r["winner_tuple"][0][0])
    return out


@torch.no_grad()
def _run_eval(
    model, build_dep_hooks, build_cln_hooks, build_lp_hooks,
    eval_dep, eval_cln, eval_cln_marker, eval_dep_lp, eval_dep_attn,
    eval_gen_dep, eval_gen_attn, clean_rollouts,
    base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device,
    use_past_kv_cache: bool = True,
):
    """Generic eval pipeline parameterised by hook builders.

    Generation reuse:
      * Steered batch gen (per eval seed) is shared by ASR, the gen-CE-ratio
        numerator, and the severity-ratio numerator — `capture_log_softmax=True`
        keeps the per-step distribution at zero extra forward cost.
      * Clean rollouts (`clean_rollouts["tokens"]`, `clean_rollouts["log_softmax"]`)
        are α-independent and are generated once in `_pregen_clean_rollouts`
        — used as the gen-CE-ratio baseline AND the severity-ratio
        denominator's distributions.

    Aggregation (matches the severity-ratio discipline):
      * `gen_ce_ratio = (Σ_{s,b,t} NLL_steered) / (Σ_{s,b,t} NLL_baseline)` —
        sums across all eval seeds × rows × generated positions, divided once.
      * `severity_ratio` — see `sleeper.metrics.severity_ratio`. Numerator
        diagonal seed pairing, denominator all unordered seed pairs, each
        side meaned independently before division.

    `eval_gen_dep` MUST equal `eval_dep_lp[: n_gen_ce]` and
    `clean_rollouts["log_softmax"]` MUST be (S, n_gen_ce, T_gen, V) keyed in
    the same `eval_seeds` order.
    """
    h_dep = build_dep_hooks()
    e_logp = teacher_forced_sleeper_logp(model, model.tokenizer, eval_dep,
                                         fwd_hooks=h_dep).mean().item()

    h_cln = build_cln_hooks()
    e_ce  = clean_continuation_ce(model, eval_cln, eval_cln_marker,
                                  fwd_hooks=h_cln).mean().item()

    h_lp = build_lp_hooks()
    n_gen_ce = eval_gen_dep.shape[0]
    asrs = []
    gen_num_sum = 0.0
    gen_den_sum = 0.0
    gen_count = 0
    steered_lsm_list: list[torch.Tensor] = []
    for s_idx, s in enumerate(eval_seeds):
        sampler = make_sampling_sampler(temperature=eval_temp, seed=int(s), device=device)
        steered_gen, steered_lsm = generate_with_hooks(
            model, eval_dep_lp, h_lp, gen_tokens, sampler,
            attention_mask=eval_dep_attn, capture_log_softmax=True,
            use_past_kv_cache=use_past_kv_cache,
        )
        asrs.append(asr_16(steered_gen, model.tokenizer))
        baseline_tokens = clean_rollouts["tokens"][s_idx][: n_gen_ce]
        r = deployment_generation_ratio(
            model, eval_gen_dep, gen_tokens=gen_tokens,
            attention_mask=eval_gen_attn,
            pre_generated_steered=steered_gen[: n_gen_ce],
            pre_generated_baseline=baseline_tokens,
        )
        gen_num_sum += r["num_sum"]
        gen_den_sum += r["den_sum"]
        gen_count   += r["count"]
        steered_lsm_list.append(steered_lsm[: n_gen_ce])

    gen_ce_ratio = gen_num_sum / max(gen_den_sum, 1e-12)

    steered_lsm_stack = torch.stack(steered_lsm_list, dim=0)              # (S, n_gen_ce, T, V) CPU fp16
    sev = severity_ratio(clean_rollouts["log_softmax"], steered_lsm_stack)

    return {
        "asr": sum(asrs) / len(asrs), "asr_per_seed": asrs,
        "delta_logp":      e_logp - base_logp,
        "delta_ce":        e_ce - base_ce,
        "gen_ce_ratio":    gen_ce_ratio,
        "gen_ce_num_mean": gen_num_sum / max(gen_count, 1),
        "gen_ce_den_mean": gen_den_sum / max(gen_count, 1),
        "severity_ratio":  sev["ratio"],
        "severity_num":    sev["num"],
        "severity_den":    sev["den"],
    }


def _upstream_eval(
    model, sae_ln1, features, alpha, W,
    eval_dep, eval_dep_pmask, eval_dep_lp, eval_dep_attn,
    eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn, clean_rollouts,
    base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device,
    use_past_kv_cache: bool = True,
):
    """Upstream: one or more ln1 SAE features steered together via OV-only
    hook (V channel, all 16 heads). `features` may be a single int (legacy
    single-feature case) or a list of ints — `resolve_channel_deltas` sums
    the per-feature deltas when multiple V-tagged features are passed in."""
    if isinstance(features, int):
        features = [features]
    sel    = [(int(f), "V") for f in features]
    active = ACTIVE_CHANNELS["ov"]
    cln_pmask = prompt_mask_from_markers(eval_cln.shape[1], eval_cln_marker.cpu()).to(device)

    def _h_dep():
        if alpha == 0.0: return []
        cd = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_dep, eval_dep_pmask)
        return build_hooks(cd, alpha, active, W, LN1_HOOK, 0)

    def _h_cln():
        if alpha == 0.0: return []
        cd = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_cln, cln_pmask)
        return build_hooks(cd, alpha, active, W, LN1_HOOK, 0)

    def _h_lp():
        if alpha == 0.0: return []
        cd = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_dep_lp, eval_dep_attn, eval_dep_attn)
        return build_hooks(cd, alpha, active, W, LN1_HOOK, 0)

    return _run_eval(model, _h_dep, _h_cln, _h_lp,
                     eval_dep, eval_cln, eval_cln_marker, eval_dep_lp, eval_dep_attn,
                     eval_gen_dep, eval_gen_attn, clean_rollouts,
                     base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device,
                     use_past_kv_cache=use_past_kv_cache)


def _downstream_eval(
    model, sae_mid, feature, alpha,
    eval_dep, eval_dep_pmask, eval_dep_lp, eval_dep_attn,
    eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn, clean_rollouts,
    base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device,
    use_past_kv_cache: bool = True,
):
    """Downstream: resid_mid SAE feature, additive hook at hook_resid_mid (no head routing)."""
    cln_pmask = prompt_mask_from_markers(eval_cln.shape[1], eval_cln_marker.cpu()).to(device)

    def _h_dep():
        if alpha == 0.0: return []
        d = compute_sae_delta(model, sae_mid, RESID_MID, feature, eval_dep, eval_dep_pmask)
        return additive_steer_hook(d, alpha, RESID_MID)

    def _h_cln():
        if alpha == 0.0: return []
        d = compute_sae_delta(model, sae_mid, RESID_MID, feature, eval_cln, cln_pmask)
        return additive_steer_hook(d, alpha, RESID_MID)

    def _h_lp():
        if alpha == 0.0: return []
        d = compute_sae_delta(model, sae_mid, RESID_MID, feature, eval_dep_lp, eval_dep_attn,
                              attention_mask=eval_dep_attn)
        return additive_steer_hook(d, alpha, RESID_MID)

    return _run_eval(model, _h_dep, _h_cln, _h_lp,
                     eval_dep, eval_cln, eval_cln_marker, eval_dep_lp, eval_dep_attn,
                     eval_gen_dep, eval_gen_attn, clean_rollouts,
                     base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device,
                     use_past_kv_cache=use_past_kv_cache)


@torch.no_grad()
def _eval_baseline_asr(
    model, eval_dep_lp, eval_dep_attn, gen_tokens, eval_seeds, eval_temp, device,
    use_past_kv_cache: bool = True,
):
    asrs = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temp, seed=int(s), device=device)
        gen = generate_with_hooks(model, eval_dep_lp, [], gen_tokens, sampler,
                                  attention_mask=eval_dep_attn,
                                  use_past_kv_cache=use_past_kv_cache)
        asrs.append(asr_16(gen, model.tokenizer))
    return sum(asrs) / len(asrs), asrs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inp", type=Path, default=Path("results/matrix_sweep.json"),
                   help="matrix_sweep.json — winners are read from its ov×ov rows.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_sel",  type=int, default=100)
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--n_gen_ce", type=int, default=25)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--sae_mid", type=Path, default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--downstream_feature", type=int, default=579,
                   help="resid_mid SAE feature ablated for the downstream-baseline curve.")
    p.add_argument("--out", type=Path, default=Path("results/single_feature_alpha_sweep.json"))
    p.add_argument("--device", default=None)
    p.add_argument("--use_past_kv_cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Use TransformerLens KV cache during sampled generation "
                        "(default on). Pass --no-use_past_kv_cache to fall back to "
                        "the full-recompute path; outputs match cache-on bit-for-bit "
                        "on tokens / ASR / token-NLL, with ~1e-7 reordering noise on "
                        "distribution-level metrics.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    winners = _ovov_winners(args.inp)
    missing = [s for s in args.seeds if s not in winners]
    if missing:
        raise SystemExit(f"no ov×ov winner in {args.inp} for seeds {missing}")

    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    sae_mid, _ = sae_load(args.sae_mid, device=device)

    splits = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                 n_test=args.n_eval, seq_len=128, seed=0)
    ev = splits["test"]
    eval_pmask = prompt_mask_from_markers(128, ev.story_marker_pos)
    eval_dep         = ev.tokens[ev.is_deployment].to(device)
    eval_dep_pmask   = eval_pmask[ev.is_deployment].to(device)
    eval_cln         = ev.tokens[~ev.is_deployment].to(device)
    eval_cln_marker  = ev.story_marker_pos[~ev.is_deployment].to(device)

    raw_dep = load_dep_prompts(tok, args.n_sel // 2 + args.n_eval // 2, split="test")
    eval_lp, eval_attn = left_pad_prompts(raw_dep[args.n_sel // 2 :
                                                    args.n_sel // 2 + args.n_eval // 2], pad_id)
    eval_lp, eval_attn = eval_lp.to(device), eval_attn.to(device)
    eval_gen_dep  = eval_lp[: args.n_gen_ce]
    eval_gen_attn = eval_attn[: args.n_gen_ce]

    base_logp = teacher_forced_sleeper_logp(model, tok, eval_dep).mean().item()
    base_ce   = clean_continuation_ce(model, eval_cln, eval_cln_marker).mean().item()
    base_asr, base_asr_per_seed = _eval_baseline_asr(
        model, eval_lp, eval_attn, args.gen_tokens, args.eval_seeds, args.eval_temperature, device,
        use_past_kv_cache=args.use_past_kv_cache,
    )
    print(f"[αsweep] eval baseline: asr={base_asr:.3f}  Δlogp_base={base_logp:.3f}  "
          f"cln_CE_base={base_ce:.4f}")

    print(f"[αsweep] pre-generating clean rollouts on stripped-clean prompts "
          f"(B={args.n_gen_ce}, S={len(args.eval_seeds)})...")
    clean_rollouts = pregen_clean_rollouts(
        model, eval_gen_dep, eval_gen_attn, args.gen_tokens,
        args.eval_seeds, args.eval_temperature, device,
        use_past_kv_cache=args.use_past_kv_cache,
    )

    points = []
    # Upstream family — per-seed ov×ov winner, V-channel hook all heads.
    for sae_seed in args.seeds:
        f = winners[sae_seed]
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{sae_seed}.pt"), device=device)
        for alpha in args.alphas:
            e = _upstream_eval(model, sae_ln1, f, alpha, W,
                               eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                               eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                               clean_rollouts,
                               base_logp, base_ce, args.gen_tokens,
                               args.eval_seeds, args.eval_temperature, device,
                               use_past_kv_cache=args.use_past_kv_cache)
            print(f"[αsweep] upstream seed={sae_seed}  f{f}  α={alpha:>4}  "
                  f"asr={e['asr']:.3f}  Δcln-CE={e['delta_ce']:+.4f}  "
                  f"gen-CE-ratio={e['gen_ce_ratio']:.3f}  "
                  f"severity={e['severity_ratio']:.3f}")
            points.append({"family": "upstream", "sae_seed": sae_seed,
                           "feature": f, "alpha": alpha, **e})

    # Downstream family — single resid_mid f579, additive hook at resid_mid.
    for alpha in args.alphas:
        e = _downstream_eval(model, sae_mid, args.downstream_feature, alpha,
                             eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                             eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                             clean_rollouts,
                             base_logp, base_ce, args.gen_tokens,
                             args.eval_seeds, args.eval_temperature, device,
                             use_past_kv_cache=args.use_past_kv_cache)
        print(f"[αsweep] downstream f{args.downstream_feature}  α={alpha:>4}  "
              f"asr={e['asr']:.3f}  Δcln-CE={e['delta_ce']:+.4f}  "
              f"gen-CE-ratio={e['gen_ce_ratio']:.3f}  "
              f"severity={e['severity_ratio']:.3f}")
        points.append({"family": "downstream", "sae_seed": None,
                       "feature": args.downstream_feature, "alpha": alpha, **e})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds, "eval_seeds": args.eval_seeds,
                                "alphas": args.alphas},
        "decoding": {"asr": {"mode": "sample", "temperature": args.eval_temperature,
                             "seeds": args.eval_seeds},
                     "gen_ce_ratio":   {"mode": "sample", "temperature": args.eval_temperature,
                                        "seeds": args.eval_seeds},
                     "severity_ratio": {"mode": "sample", "temperature": args.eval_temperature,
                                        "seeds": args.eval_seeds}},
        "baseline": {"asr": base_asr, "asr_per_seed": base_asr_per_seed,
                     "dep_logp": base_logp, "clean_ce": base_ce},
        "winners": winners,
        "points": points,
    }, indent=2, default=str))
    print(f"\n[αsweep] wrote {args.out}")


if __name__ == "__main__":
    main()
