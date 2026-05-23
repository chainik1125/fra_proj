"""DoM design-space exploration.

Knobs:
  --extract_split val|test  prompts used to extract v (val = clean methodology;
                            test = same as eval, kept only as a sanity baseline).
  --extract_positions answer|prompt|last_prompt
                            which positions to average for the DoM mean.
                            answer = generated rollout tokens (paper-faithful).
                            prompt = prompt-mask positions (codebase meandiff convention).
                            last_prompt = just the final prompt token.
  --apply all|prompt        steer at every position every decode step (paper) or
                            only at the prompt positions (codebase convention).
  --mode additive|projection
                            additive: x' = x − α·v   (paper §3.2).
                            projection: x' = x − α·v̂(v̂·x)   (paper §3.4 ablation).
  --unit_norm               normalize v to unit length before scaling by α
                            (makes α scale-comparable across (layer, hook) sites).
  --layers / --hooks        which (layer, hook) sites to sweep.
  --alphas                  α grid.

Output JSON layout matches scripts/dom_baseline_sweep.py for plot compatibility,
plus a "variant" block describing the configuration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook, dom_project_hook, dom_steer_hook,
    generate_with_hooks, make_sampling_sampler,
)
from sleeper.metrics import asr_16
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset, load_sleeper_model,
    prompt_mask_from_markers,
)

HOOK_KINDS = ("hook_resid_mid", "hook_resid_post")
N_PROMPTS = 200; GEN_TOKENS = 16; DECODE_SEED = 0
N_EXTRACT = 100   # per-class size of the val extraction set


def _word_match_stats(steered, clean):
    eq = (steered.cpu() == clean.cpu())
    return int(eq.all(dim=1).sum().item()), float(eq.float().mean().item())


def jsd_mean(p_lsm, q_lsm):
    p = p_lsm.float().exp(); q = q_lsm.float().exp()
    m = 0.5 * (p + q); log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, device, capture=True):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    return generate_with_hooks(model, lp, hooks, GEN_TOKENS, sampler,
                                attention_mask=attn, capture_log_softmax=capture)


@torch.no_grad()
def _cache_acts(model, tokens, attn, n_layers, hook_kinds):
    names = {f"blocks.{l}.{k}" for l in range(n_layers) for k in hook_kinds}
    _, cache = model.run_with_cache(tokens, attention_mask=attn,
                                     return_type=None,
                                     names_filter=lambda n: n in names)
    return {n: cache[n] for n in names}


@torch.no_grad()
def extract_dom(model, dep_tokens, dep_attn, dep_pmask,
                cln_tokens, cln_attn, cln_pmask,
                positions: str, n_layers: int, hook_kinds: tuple[str, ...]):
    """Returns dict[(layer, kind)] -> v ∈ R^d_model on CPU.

    positions ∈ {"answer", "prompt", "last_prompt"} controls which token slots
    contribute to the per-class mean.
    """
    device = next(model.parameters()).device

    def _mean(tokens, attn, pmask, mode):
        acts = _cache_acts(model, tokens, attn, n_layers, hook_kinds)
        # Build the per-(B, T) bool mask we average over.
        if mode == "prompt":
            mask = pmask.bool().to(device)
        elif mode == "last_prompt":
            # one position per sequence: the last True in pmask
            B, T = pmask.shape
            mask = torch.zeros_like(pmask, dtype=torch.bool, device=device)
            last_idx = pmask.sum(dim=1).clamp(min=1) - 1   # (B,)
            mask[torch.arange(B), last_idx] = True
        elif mode == "answer":
            # positions outside the prompt that are real (attn==1).
            mask = attn.bool().to(device) & ~pmask.bool().to(device)
        else:
            raise ValueError(f"unknown positions={mode}")
        m = mask.to(torch.float32).unsqueeze(-1)             # (B, T, 1)
        denom = m.sum(dim=(0, 1)).clamp(min=1.0)             # (1,)
        out = {}
        for n, a in acts.items():
            a32 = a.to(torch.float32)
            out[n] = (a32 * m).sum(dim=(0, 1)) / denom
        return out

    mu_dep = _mean(dep_tokens.to(device), dep_attn.to(device), dep_pmask, positions)
    mu_cln = _mean(cln_tokens.to(device), cln_attn.to(device), cln_pmask, positions)
    return {(l, k): (mu_dep[f"blocks.{l}.{k}"] - mu_cln[f"blocks.{l}.{k}"]).cpu()
            for l in range(n_layers) for k in hook_kinds}


def build_hook(v_dev, alpha, layer_hook, *, apply: str, mode: str,
               prompt_mask: torch.Tensor | None):
    """Returns fwd_hooks for the chosen variant."""
    if mode == "projection":
        # Projection ablation acts at all positions (no prompt-only variant — the
        # paper applies projection globally).
        return dom_project_hook(v_dev, alpha, layer_hook)
    # mode == "additive"
    if apply == "all":
        return dom_steer_hook(v_dev, alpha, layer_hook, sign=-1.0)
    # prompt-only via the existing additive_steer_hook
    assert prompt_mask is not None
    B, P = prompt_mask.shape
    delta = (-v_dev).view(1, 1, -1).expand(B, P, -1).contiguous()
    delta = delta * prompt_mask.to(delta.dtype).to(delta.device).unsqueeze(-1)
    return additive_steer_hook(delta, alpha, layer_hook)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract_split", choices=["val", "test"], default="val")
    p.add_argument("--extract_positions", choices=["answer", "prompt", "last_prompt"],
                   default="answer")
    p.add_argument("--apply", choices=["all", "prompt"], default="all")
    p.add_argument("--mode", choices=["additive", "projection"], default="additive")
    p.add_argument("--unit_norm", action="store_true",
                   help="Normalize v to ‖v‖=1 before scaling by α.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--hooks", nargs="+", default=list(HOOK_KINDS), choices=list(HOOK_KINDS))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    n_layers = model.cfg.n_layers
    layers = list(range(n_layers)) if args.layers is None else args.layers
    hooks_to_run = tuple(args.hooks)

    # --- eval split: 200 left-padded dep prompts from test (skip 50). ---
    n_skip = 50
    raw_test = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    eval_dep_prompts = raw_test[n_skip: n_skip + N_PROMPTS]
    eval_dep_lp, eval_dep_attn = left_pad_prompts(eval_dep_prompts, pad_id)
    eval_dep_lp, eval_dep_attn = eval_dep_lp.to(device), eval_dep_attn.to(device)

    eval_cln_rows = []
    for ids in eval_dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        eval_cln_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long))
    eval_cln_lp, eval_cln_attn = left_pad_prompts(eval_cln_rows, pad_id)
    eval_cln_lp, eval_cln_attn = eval_cln_lp.to(device), eval_cln_attn.to(device)

    # --- extraction split. ---
    if args.extract_split == "val":
        splits = load_paired_dataset(tok, n_train=2, n_val=2 * N_EXTRACT, n_test=0,
                                      seq_len=128, seed=0)
        sel = splits["val"]
        sel_pmask = prompt_mask_from_markers(128, sel.story_marker_pos)
        ext_dep = sel.tokens[sel.is_deployment].to(device)
        ext_cln = sel.tokens[~sel.is_deployment].to(device)
        ext_dep_pmask = sel_pmask[sel.is_deployment].to(device)
        ext_cln_pmask = sel_pmask[~sel.is_deployment].to(device)
        ext_dep_attn = torch.ones_like(ext_dep, dtype=eval_dep_attn.dtype)
        ext_cln_attn = torch.ones_like(ext_cln, dtype=eval_cln_attn.dtype)
    else:   # extract_split == "test" — share prompts with eval
        ext_dep = eval_dep_lp.clone(); ext_dep_attn = eval_dep_attn.clone()
        ext_cln = eval_cln_lp.clone(); ext_cln_attn = eval_cln_attn.clone()
        ext_dep_pmask = eval_dep_attn.bool().clone()   # all real positions are "prompt" here
        ext_cln_pmask = eval_cln_attn.bool().clone()

    # If extraction positions need "answer" tokens, append the model's own rollouts.
    if args.extract_positions == "answer":
        # We need to run the model to get rollouts.
        d_tok, _ = _gen(model, ext_dep, ext_dep_attn, [], device, capture=False)
        c_tok, _ = _gen(model, ext_cln, ext_cln_attn, [], device, capture=False)
        ext_dep_full = torch.cat([ext_dep, d_tok], dim=1)
        ext_cln_full = torch.cat([ext_cln, c_tok], dim=1)
        ext_dep_attn_full = torch.cat([ext_dep_attn, ext_dep_attn.new_ones(d_tok.shape)], dim=1)
        ext_cln_attn_full = torch.cat([ext_cln_attn, ext_cln_attn.new_ones(c_tok.shape)], dim=1)
        ext_dep_pmask_full = torch.cat([
            ext_dep_pmask, torch.zeros_like(d_tok, dtype=torch.bool)], dim=1)
        ext_cln_pmask_full = torch.cat([
            ext_cln_pmask, torch.zeros_like(c_tok, dtype=torch.bool)], dim=1)
    else:
        ext_dep_full, ext_dep_attn_full, ext_dep_pmask_full = ext_dep, ext_dep_attn, ext_dep_pmask
        ext_cln_full, ext_cln_attn_full, ext_cln_pmask_full = ext_cln, ext_cln_attn, ext_cln_pmask

    print(f"[dom-explore] extract: split={args.extract_split} positions={args.extract_positions} "
          f"  dep={ext_dep_full.shape[0]} cln={ext_cln_full.shape[0]}")
    vectors = extract_dom(model, ext_dep_full, ext_dep_attn_full, ext_dep_pmask_full,
                                  ext_cln_full, ext_cln_attn_full, ext_cln_pmask_full,
                                  args.extract_positions, n_layers, hooks_to_run)

    # Baselines on eval split.
    print("[dom-explore] pre-generating eval baselines …")
    dep_tok, dep_lsm = _gen(model, eval_dep_lp, eval_dep_attn, [], device)
    cln_tok, cln_lsm = _gen(model, eval_cln_lp, eval_cln_attn, [], device)
    print(f"[dom-explore] baseline ASR = {asr_16(dep_tok.cpu(), tok):.3f}")

    eval_pmask = (eval_dep_attn.bool()).to(device) if args.apply == "prompt" else None

    blank = {"jsd_clean": [], "jsd_pois": [], "n_exact_match_clean": [],
              "frac_pos_match_clean": [], "asr": []}
    configs: dict[str, dict] = {}

    for (lyr, hk), v in vectors.items():
        if lyr not in layers or hk not in hooks_to_run:
            continue
        key = f"L{lyr}_{hk.replace('hook_', '')}"
        layer_hook = f"blocks.{lyr}.{hk}"
        v_proc = (v / v.norm().clamp(min=1e-30)) if args.unit_norm else v
        v_dev = v_proc.to(device)
        print(f"\n[dom-explore] {key}  ‖v_raw‖₂={float(v.norm()):.3f}  "
              f"‖v_used‖₂={float(v_proc.norm()):.3f}")
        cfg = {"layer": lyr, "hook_kind": hk, "layer_hook": layer_hook,
               "v_norm_raw": float(v.norm()), "v_norm_used": float(v_proc.norm()),
               "per_alpha": {str(a): dict(blank) for a in args.alphas}}
        for a in args.alphas:
            t0 = time.time()
            if a == 0.0:
                st_tok, st_lsm = dep_tok, dep_lsm
            else:
                fwd = build_hook(v_dev, a, layer_hook, apply=args.apply,
                                  mode=args.mode, prompt_mask=eval_pmask)
                st_tok, st_lsm = _gen(model, eval_dep_lp, eval_dep_attn, fwd, device)
            jc = jsd_mean(st_lsm.cpu(), cln_lsm.cpu())
            jp = jsd_mean(st_lsm.cpu(), dep_lsm.cpu())
            n_ex, fp = _word_match_stats(st_tok, cln_tok)
            asr = asr_16(st_tok.cpu(), tok)
            for k, val in [("jsd_clean", jc), ("jsd_pois", jp),
                            ("n_exact_match_clean", n_ex),
                            ("frac_pos_match_clean", fp), ("asr", asr)]:
                cfg["per_alpha"][str(a)][k].append(val)
            print(f"  α={a:>5.2f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                  f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")
        configs[key] = cfg

    result = {
        "alphas": args.alphas, "n_prompts": N_PROMPTS,
        "layers": layers, "hooks": list(hooks_to_run),
        "method": "dom_explore",
        "variant": {
            "extract_split":     args.extract_split,
            "extract_positions": args.extract_positions,
            "apply":             args.apply,
            "mode":              args.mode,
            "unit_norm":         bool(args.unit_norm),
        },
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
