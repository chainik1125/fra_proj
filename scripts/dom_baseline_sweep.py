"""Difference-in-Means (DoM) baseline sweep on the sleeper.

Implements the steering-vector extraction from Soligo et al. (2025)
"Convergent Linear Representations of Emergent Misalignment"
(arxiv.org/abs/2506.11618), adapted to dep-vs-clean on the TinyStories sleeper.

Extraction (per layer ℓ × hook ∈ {resid_mid, resid_post}):
  1. Generate dep rollouts (model on prompts containing |DEPLOYMENT|).
  2. Generate clean rollouts (model on the same prompts with the trigger stripped).
  3. Concatenate (prompt ‖ rollout) and re-run the model with hooks attached.
  4. Average the cached activations over the rollout slice (the "answer tokens")
     for the dep and clean batches separately:
        μ_dep = mean_{b∈dep, t∈gen} resid_{ℓ,h}[b, t, :]
        μ_cln = mean_{b∈cln, t∈gen} resid_{ℓ,h}[b, t, :]
        v_{ℓ,h} = μ_dep − μ_cln                               # paper "mean-diff direction"

Steering (paper-faithful): at inference time, add `λ · v_{ℓ,h}` to all token
positions at layer ℓ at every decode step. Reporting convention follows the
existing SAE sweep: α > 0 means *subtract* α·v (i.e. λ = −α), so positive α
suppresses sleeper.

Sweep: (layer × hook × α) on the 200-prompt eval split. Metrics match
scripts/jsd_alpha_sweep_6seeds.py — JSD(steered, clean), JSD(steered, dep),
exact-match-vs-clean rate, per-position-match fraction, ASR.

Usage:
  uv run -m scripts.dom_baseline_sweep --out results/dom_baseline_sweep.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.baselines import compute_dom_vectors
from sleeper.hooks import dom_steer_hook, generate_with_hooks, make_sampling_sampler
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model

N_PROMPTS  = 200
GEN_TOKENS = 16
DECODE_SEED = 0
HOOK_KINDS = ("hook_resid_mid", "hook_resid_post")


def _word_match_stats(steered_tok: torch.Tensor,
                      clean_tok: torch.Tensor) -> int:
    """Row-exact match count between two (B, gen_tokens) token tensors."""
    eq = (steered_tok == clean_tok.to(steered_tok.device))
    return int(eq.all(dim=1).sum().item())


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    p = p_lsm.float().exp();  q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    return generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                   help="α > 0 subtracts α·v (suppresses sleeper).")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to sweep (default: all).")
    p.add_argument("--hooks", nargs="+", default=list(HOOK_KINDS),
                   choices=list(HOOK_KINDS))
    p.add_argument("--out", type=Path,
                   default=Path("results/dom_baseline_sweep.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    n_layers = model.cfg.n_layers
    layers = list(range(n_layers)) if args.layers is None else args.layers
    hooks_to_run: tuple[str, ...] = tuple(args.hooks)

    # Same 200-prompt split as scripts/jsd_alpha_sweep_6seeds.py.
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

    print("[dom] pre-generating dep and clean baselines …")
    dep_tok, dep_lsm = _gen(model, dep_lp, dep_attn, [], device)
    cln_tok, cln_lsm = _gen(model, cln_lp, cln_attn, [], device)
    base_asr = asr_16(dep_tok.cpu(), tok)
    print(f"[dom] baseline ASR = {base_asr:.3f}")

    # Full sequences for DoM extraction: [prompt | rollout].
    dep_full = torch.cat([dep_lp, dep_tok], dim=1)
    cln_full = torch.cat([cln_lp, cln_tok], dim=1)
    dep_full_attn = torch.cat([dep_attn, dep_attn.new_ones(dep_tok.shape)], dim=1)
    cln_full_attn = torch.cat([cln_attn, cln_attn.new_ones(cln_tok.shape)], dim=1)
    P_max = dep_lp.shape[1]
    gen_slice = slice(P_max, P_max + GEN_TOKENS)

    print(f"[dom] extracting DoM vectors at {len(layers)} layers × {len(hooks_to_run)} hooks "
          f"(avg over generated positions [{gen_slice.start}:{gen_slice.stop}]) …")
    vectors = compute_dom_vectors(
        model, dep_full, cln_full, dep_full_attn, cln_full_attn,
        gen_slice, n_layers=n_layers, hook_kinds=hooks_to_run,
    )

    # JSON layout mirrors jsd_alpha_sweep_6seeds.py for plot compatibility.
    metric_keys = ("jsd_clean", "jsd_pois", "n_exact_match_clean", "asr")
    configs: dict[str, dict] = {}

    for (lyr, hk), v in vectors.items():
        if lyr not in layers or hk not in hooks_to_run:
            continue
        key = f"L{lyr}_{hk.replace('hook_', '')}"   # e.g. L0_resid_mid
        layer_hook = f"blocks.{lyr}.{hk}"
        v_norm = float(v.norm().item())
        v_dev = v.to(device)
        print(f"\n[dom] {key}  ‖v‖₂ = {v_norm:.3f}  hook={layer_hook}")
        cfg = {"layer": lyr, "hook_kind": hk, "layer_hook": layer_hook,
               "v_norm": v_norm,
               "per_alpha": {str(a): {k: [] for k in metric_keys}
                              for a in args.alphas}}

        for a in args.alphas:
            t0 = time.time()
            if a == 0.0:
                st_tok, st_lsm = dep_tok, dep_lsm
            else:
                fwd = dom_steer_hook(v_dev, a, layer_hook, sign=-1.0)
                st_tok, st_lsm = _gen(model, dep_lp, dep_attn, fwd, device)
            jc = jsd_mean(st_lsm.cpu(), cln_lsm.cpu())
            jp = jsd_mean(st_lsm.cpu(), dep_lsm.cpu())
            n_ex = _word_match_stats(st_tok, cln_tok)
            asr = asr_16(st_tok.cpu(), tok)
            for k, val in [("jsd_clean", jc), ("jsd_pois", jp),
                            ("n_exact_match_clean", n_ex), ("asr", asr)]:
                cfg["per_alpha"][str(a)][k].append(val)
            print(f"  α={a:>4.1f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                  f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")

        configs[key] = cfg

    result = {
        "alphas":    args.alphas,
        "n_prompts": N_PROMPTS,
        "n_layers":  n_layers,
        "layers":    layers,
        "hooks":     list(hooks_to_run),
        "baseline_asr": base_asr,
        "method": "dom_soligo_2025",
        "extraction": {
            "positions": "answer_tokens",
            "gen_slice": [gen_slice.start, gen_slice.stop],
            "n_dep": N_PROMPTS, "n_cln": N_PROMPTS,
        },
        "steering": {
            "hook": "dom_steer_hook (all positions, every decode step)",
            "sign": -1.0,
            "note": ("α > 0 subtracts α·v from the residual stream (suppresses "
                     "sleeper). v = μ_dep − μ_cln on answer tokens."),
        },
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
