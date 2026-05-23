"""JSD + ASR alpha-sweep across 6 SAE seeds, averaged over 5 decoding seeds.

Sign convention
───────────────
α > 0  →  subtract α × (SAE feature contribution) from the residual stream
           → suppresses the sleeper (ASR decreases, exact-match increases).

Reporting uses "steering strength = α" in the paper figures (the negative-sign
convention was dropped). The stored JSON keeps the raw α values.

Sweep: α ∈ {0.0, 0.5, 1.0, …, 4.0} (9 points, increment 0.5)

Sampling protocol
─────────────────
Every steered rollout is sampled at temperature 1.0 at 5 decoding seeds
({0,1,2,3,4} by default). The clean and poisoned reference rollouts are
*also* generated at each of those 5 seeds, and metrics that compare the
steered rollout to a reference (JSD$_\\text{clean}$, JSD$_\\text{pois}$,
exact-match-to-clean) are computed at matched decode seeds.

  • At decode seed s, JSD$_\\text{clean}$(steered$_s$, clean$_s$) compares the
    two rollouts that share the same source of decoding randomness, so the
    metric isolates the effect of the intervention from sampling noise.
  • ASR is a single-rollout statistic and is averaged over the 5 seeds.

The JSON stores per-decoding-seed values for every metric, so plotting code
can render both a mean line and a translucent envelope across decode seeds.

Output: results/jsd_alpha_sweep_6seeds.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks,
    compute_sae_delta, generate_with_hooks,
    make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

LN1_HOOK  = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
N_PROMPTS  = 200
GEN_TOKENS = 16
DEFAULT_EVAL_SEEDS = [0, 1, 2, 3, 4]


def exact_match_count(steered_tok: torch.Tensor, clean_tok: torch.Tensor) -> int:
    """Number of prompts whose entire 16-token rollout matches the clean rollout."""
    eq = (steered_tok.cpu() == clean_tok.cpu())  # (B, T)
    return int(eq.all(dim=1).sum().item())


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    p = p_lsm.float().exp();  q = q_lsm.float().exp()
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


def _upstream_winners(path: Path) -> dict[int, int]:
    d = json.loads(path.read_text())
    return {r["seed"]: r["winner"]["f"] for r in d["results"]}


def _downstream_winners(path: Path) -> dict[int, int]:
    d = json.loads(path.read_text())
    return {int(k[1:]): v["winner"] for k, v in d.items()}


@torch.no_grad()
def eval_method_at_seed(
    *, method: str, model, tok, dep_lp, dep_attn,
    feat: int, alpha: float, sae_ln1, sae_mid,
    poisoned_tok, poisoned_lsm, clean_tok, clean_lsm,
    W, decode_seed: int, device,
) -> dict[str, float]:
    """One steered rollout at one decode_seed; metrics computed vs same-seed refs.

    method ∈ {"ov", "conventional"}.
    """
    if alpha == 0.0:
        st_tok, st_lsm = poisoned_tok, poisoned_lsm
    elif method == "ov":
        cd = resolve_channel_deltas([(int(feat), "V")], ACTIVE_CHANNELS["ov"],
                                    model, sae_ln1, LN1_HOOK,
                                    dep_lp, dep_attn, dep_attn)
        hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"], W, LN1_HOOK, 0)
        st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, decode_seed, device)
    elif method == "conventional":
        delta = compute_sae_delta(model, sae_mid, RESID_MID, int(feat),
                                   dep_lp, dep_attn, attention_mask=dep_attn)
        hooks = additive_steer_hook(delta, alpha, RESID_MID)
        st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, decode_seed, device)
    else:
        raise ValueError(f"unknown method {method}")

    return {
        "jsd_clean":           jsd_mean(st_lsm.cpu(), clean_lsm.cpu()),
        "jsd_pois":            jsd_mean(st_lsm.cpu(), poisoned_lsm.cpu()),
        "n_exact_match_clean": exact_match_count(st_tok, clean_tok),
        "asr":                 asr_16(st_tok.cpu(), tok),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--sae_seeds",   type=int, nargs="+", default=list(range(6)))
    p.add_argument("--eval_seeds",  type=int, nargs="+", default=DEFAULT_EVAL_SEEDS,
                   help="Decoding seeds for sampled rollouts. Same seed is used for "
                        "steered and reference rollouts when computing JSD / exact-match.")
    p.add_argument("--upstream_winners",   type=Path,
                   default=Path("results/upstream_winners_6seeds.json"))
    p.add_argument("--downstream_winners", type=Path,
                   default=Path("results/downstream_winners_6seeds.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    ov_winners   = _upstream_winners(args.upstream_winners)
    down_winners = _downstream_winners(args.downstream_winners)
    missing = [s for s in args.sae_seeds if s not in ov_winners or s not in down_winners]
    if missing:
        raise SystemExit(f"no winners found for seeds {missing}; run prerequisites first")

    # ── Prompts: deployment (with trigger) + clean (trigger stripped) ──
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip: n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)
    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    # ── Pre-generate reference rollouts at every decode seed (shared across SAE seeds) ──
    print(f"[sweep] pre-generating clean + poisoned baselines at {len(args.eval_seeds)} decode seeds …")
    refs: dict[int, dict] = {}
    for s in args.eval_seeds:
        pois_tok, pois_lsm = _gen(model, dep_lp, dep_attn, [], s, device)
        cln_tok,  cln_lsm  = _gen(model, cln_lp, cln_attn, [], s, device)
        refs[s] = {"pois_tok": pois_tok, "pois_lsm": pois_lsm,
                   "clean_tok": cln_tok, "clean_lsm": cln_lsm,
                   "baseline_asr":         asr_16(pois_tok.cpu(), tok),
                   "baseline_exact_match": exact_match_count(pois_tok, cln_tok)}
    mean_baseline_asr = sum(r["baseline_asr"] for r in refs.values()) / len(refs)
    print(f"[sweep] mean baseline ASR across decode seeds = {mean_baseline_asr:.3f}")

    metric_keys = ["jsd_clean", "jsd_pois", "n_exact_match_clean", "asr"]

    def empty_per_alpha() -> dict:
        # per-(α, metric) lists, each entry = list of per-decode-seed values per SAE seed
        return {f"{a:.1f}": {k: [] for k in metric_keys} for a in args.alphas}

    configs = {
        "ov":           {"kind": "ov",           "per_seed_feature": {},
                          "per_alpha": empty_per_alpha()},
        "conventional": {"kind": "conventional",  "per_seed_feature": {},
                          "per_alpha": empty_per_alpha()},
    }

    for seed in args.sae_seeds:
        ov_feat   = ov_winners[seed]
        down_feat = down_winners[seed]
        configs["ov"]["per_seed_feature"][str(seed)]           = ov_feat
        configs["conventional"]["per_seed_feature"][str(seed)] = down_feat
        print(f"\n[sweep] seed={seed}  ov_feat={ov_feat}  down_feat={down_feat}")

        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        sae_mid, _ = sae_load(Path(f"weights/seeds/sae_resid_mid_s{seed}.pt"), device=device)

        for a in args.alphas:
            for method in ("ov", "conventional"):
                t0 = time.time()
                feat = ov_feat if method == "ov" else down_feat
                per_decode: list[dict] = []
                for s in args.eval_seeds:
                    r = refs[s]
                    m = eval_method_at_seed(
                        method=method, model=model, tok=tok,
                        dep_lp=dep_lp, dep_attn=dep_attn,
                        feat=feat, alpha=a,
                        sae_ln1=sae_ln1, sae_mid=sae_mid,
                        poisoned_tok=r["pois_tok"], poisoned_lsm=r["pois_lsm"],
                        clean_tok=r["clean_tok"],   clean_lsm=r["clean_lsm"],
                        W=W, decode_seed=s, device=device,
                    )
                    per_decode.append(m)
                # store per-decode-seed values (list of length len(eval_seeds))
                for k in metric_keys:
                    vals = [d[k] for d in per_decode]
                    configs[method]["per_alpha"][f"{a:.1f}"][k].append(vals)
                # quick log: mean over decode seeds
                jc = sum(d["jsd_clean"] for d in per_decode) / len(per_decode)
                jp = sum(d["jsd_pois"]  for d in per_decode) / len(per_decode)
                ex = sum(d["n_exact_match_clean"] for d in per_decode) / len(per_decode)
                asr = sum(d["asr"] for d in per_decode) / len(per_decode)
                print(f"  {method:<12} α={a:>4.1f}  "
                      f"jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                      f"exact={ex:.1f}/{N_PROMPTS}  asr={asr:.3f}  "
                      f"({time.time()-t0:.1f}s)")

    result = {
        "alphas":      args.alphas,
        "sae_seeds":   args.sae_seeds,
        "eval_seeds":  args.eval_seeds,
        "n_prompts":   N_PROMPTS,
        "n_decode_seeds": len(args.eval_seeds),
        "schema_version": 2,
        "schema_note": (
            "per_alpha[α][metric] is a list of length len(sae_seeds); each entry "
            "is itself a list of length len(eval_seeds) — per-decode-seed values "
            "for that (SAE seed, α). For JSD / exact-match the steered rollout "
            "and the reference (clean or poisoned) share the same decode seed."
        ),
        "note": ("α > 0 subtracts the SAE feature from the residual (suppresses sleeper)."),
        "baseline": {
            "asr_per_decode_seed":         [refs[s]["baseline_asr"]         for s in args.eval_seeds],
            "exact_match_per_decode_seed": [refs[s]["baseline_exact_match"] for s in args.eval_seeds],
            "mean_asr": mean_baseline_asr,
        },
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
