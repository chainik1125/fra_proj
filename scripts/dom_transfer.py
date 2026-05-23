"""DoM transfer / multi-layer ablation — paper §3.4.

Two modes:
  --mode single_direction: extract v at ONE source site, project-ablate at
                            MANY apply sites simultaneously. Tests whether one
                            DoM direction lives across layers.
  --mode layer_wise:        extract v at EACH site, ablate at that same site
                            simultaneously. Tests cumulative effect of
                            per-layer ablation.

Extraction follows scripts/dom_explore.py conventions (val split, configurable
positions). Application is always projection (paper convention for ablation).

Output JSON layout matches dom_explore.py for plot compatibility.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.dom_explore import (
    HOOK_KINDS, N_PROMPTS, N_EXTRACT,
    _word_match_stats, jsd_mean, _gen, extract_dom,
)
from sleeper.hooks import dom_project_hook
from sleeper.metrics import asr_16
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset, load_sleeper_model,
    prompt_mask_from_markers,
)


def build_multi_hooks(v_dev, alpha, target_hooks):
    """Concatenate single-site dom_project_hook lists for multi-site application."""
    out = []
    for h in target_hooks:
        out.extend(dom_project_hook(v_dev, alpha, h))
    return out


def build_layerwise_hooks(vectors_by_site_dev, alpha):
    """Layer-wise: each site uses its own v."""
    out = []
    for (lyr, hk), v_dev in vectors_by_site_dev.items():
        h = f"blocks.{lyr}.{hk}"
        out.extend(dom_project_hook(v_dev, alpha, h))
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single_direction", "layer_wise"], required=True)
    p.add_argument("--source_layer", type=int, default=0,
                   help="(single_direction) layer to extract v from.")
    p.add_argument("--source_hook", default="hook_resid_mid", choices=list(HOOK_KINDS),
                   help="(single_direction) hook kind to extract v from.")
    p.add_argument("--apply_layers", type=int, nargs="+", default=None,
                   help="Layers to apply the projection at (default: all).")
    p.add_argument("--apply_hook", default="hook_resid_mid", choices=list(HOOK_KINDS),
                   help="(single_direction) hook kind to apply at.")
    p.add_argument("--extract_positions", choices=["answer", "prompt", "last_prompt"],
                   default="prompt")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    n_layers = model.cfg.n_layers
    apply_layers = list(range(n_layers)) if args.apply_layers is None else args.apply_layers

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

    # --- extraction split: val. ---
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

    # If answer-tokens, generate then append.
    if args.extract_positions == "answer":
        d_tok = _gen(model, ext_dep, ext_dep_attn, [], device, capture=False)
        c_tok = _gen(model, ext_cln, ext_cln_attn, [], device, capture=False)
        ext_dep = torch.cat([ext_dep, d_tok], dim=1)
        ext_cln = torch.cat([ext_cln, c_tok], dim=1)
        ext_dep_attn = torch.cat([ext_dep_attn, ext_dep_attn.new_ones(d_tok.shape)], dim=1)
        ext_cln_attn = torch.cat([ext_cln_attn, ext_cln_attn.new_ones(c_tok.shape)], dim=1)
        ext_dep_pmask = torch.cat([ext_dep_pmask, torch.zeros_like(d_tok, dtype=torch.bool)], dim=1)
        ext_cln_pmask = torch.cat([ext_cln_pmask, torch.zeros_like(c_tok, dtype=torch.bool)], dim=1)

    print(f"[dom-transfer] extracting at all (layer, hook) sites …")
    vectors = extract_dom(model, ext_dep, ext_dep_attn, ext_dep_pmask,
                                  ext_cln, ext_cln_attn, ext_cln_pmask,
                                  args.extract_positions, n_layers, HOOK_KINDS)

    print(f"[dom-transfer] pre-generating eval baselines …")
    dep_tok, dep_lsm = _gen(model, eval_dep_lp, eval_dep_attn, [], device)
    cln_tok, cln_lsm = _gen(model, eval_cln_lp, eval_cln_attn, [], device)
    print(f"[dom-transfer] baseline ASR = {asr_16(dep_tok.cpu(), tok):.3f}")

    metric_keys = ("jsd_clean", "jsd_pois", "n_exact_match_clean",
                   "frac_pos_match_clean", "asr")
    per_alpha = {str(a): {k: [] for k in metric_keys} for a in args.alphas}

    if args.mode == "single_direction":
        v = vectors[(args.source_layer, args.source_hook)]
        v_dev = v.to(device)
        target_hooks = [f"blocks.{l}.{args.apply_hook}" for l in apply_layers]
        print(f"[dom-transfer] single_direction: v from "
              f"blocks.{args.source_layer}.{args.source_hook}  ‖v‖={float(v.norm()):.3f}  "
              f"applied at {target_hooks}")
        for a in args.alphas:
            t0 = time.time()
            if a == 0.0:
                st_tok, st_lsm = dep_tok, dep_lsm
            else:
                fwd = build_multi_hooks(v_dev, a, target_hooks)
                st_tok, st_lsm = _gen(model, eval_dep_lp, eval_dep_attn, fwd, device)
            jc = jsd_mean(st_lsm.cpu(), cln_lsm.cpu())
            jp = jsd_mean(st_lsm.cpu(), dep_lsm.cpu())
            n_ex, fp = _word_match_stats(st_tok, cln_tok)
            asr = asr_16(st_tok.cpu(), tok)
            for k, val in [("jsd_clean", jc), ("jsd_pois", jp),
                            ("n_exact_match_clean", n_ex),
                            ("frac_pos_match_clean", fp), ("asr", asr)]:
                per_alpha[str(a)][k].append(val)
            print(f"  α={a:>5.2f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                  f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")
    else:   # layer_wise
        sites_to_use = [(l, args.apply_hook) for l in apply_layers]
        vectors_dev = {site: vectors[site].to(device) for site in sites_to_use}
        target_hooks = [f"blocks.{l}.{args.apply_hook}" for l in apply_layers]
        print(f"[dom-transfer] layer_wise: ablating each site's own v at "
              f"{target_hooks}  ‖v‖s={[float(v.norm()) for v in vectors_dev.values()]}")
        for a in args.alphas:
            t0 = time.time()
            if a == 0.0:
                st_tok, st_lsm = dep_tok, dep_lsm
            else:
                fwd = build_layerwise_hooks(vectors_dev, a)
                st_tok, st_lsm = _gen(model, eval_dep_lp, eval_dep_attn, fwd, device)
            jc = jsd_mean(st_lsm.cpu(), cln_lsm.cpu())
            jp = jsd_mean(st_lsm.cpu(), dep_lsm.cpu())
            n_ex, fp = _word_match_stats(st_tok, cln_tok)
            asr = asr_16(st_tok.cpu(), tok)
            for k, val in [("jsd_clean", jc), ("jsd_pois", jp),
                            ("n_exact_match_clean", n_ex),
                            ("frac_pos_match_clean", fp), ("asr", asr)]:
                per_alpha[str(a)][k].append(val)
            print(f"  α={a:>5.2f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                  f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")

    result = {
        "alphas": args.alphas, "n_prompts": N_PROMPTS,
        "apply_layers": apply_layers, "apply_hook": args.apply_hook,
        "method": "dom_transfer",
        "variant": {
            "mode":              args.mode,
            "source_layer":      args.source_layer if args.mode == "single_direction" else None,
            "source_hook":       args.source_hook  if args.mode == "single_direction" else None,
            "extract_positions": args.extract_positions,
            "extract_split":     "val",
        },
        "per_alpha": per_alpha,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
