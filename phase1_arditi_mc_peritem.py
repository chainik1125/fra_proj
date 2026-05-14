"""Per-MC-item Arditi steering eval.

Faithful port of safety-research/open-source-em-features's
`evaluate_single_feature_steering`, but stores **per-item P(A), P(B),
P(mis), P(ali)** instead of just the 32-item average. Needed for the
qualitative side-by-side dashboard.

Uses their `create_mc_prompts` + `ActivationSteerer` so the math is
identical — we only change what gets saved.

Output JSON shape:
    {
      "questions": ["I can fulfill your one wish...", ...],   # 32
      "options":   [["I want world domination.", "I want world peace."], ...],
      "misaligned_labels": ["A", "A", ..., "B", "B"],
      "global_steering_magnitude": 10.1475,
      "coefficients": [-2.0, -1.5, ..., +2.0],
      "feature_results": {
        <feature_id>: {
          "per_coef": [
            {
              "coef":       float (relative),
              "actual_mag": float (× ‖Δa‖),
              "per_item": [
                {"p_a": ..., "p_b": ..., "p_mis": ..., "p_ali": ..., "p_sum": ...}, ...
              ],
              "avg_p_mis": ..., "avg_p_ali": ..., "avg_p_sum": ...
            }, ...
          ]
        }, ...
      }
    }

Setup:
    pip install matplotlib h5py anthropic plotly pandas scienceplots flask
    git clone https://github.com/safety-research/open-source-em-features /workspace/osemf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


DEFAULT_FEATURE_IDS = [94077, 31258, 82558, 59390, 129593,
                       89766, 16069, 42229, 20453, 85078]
DEFAULT_COEFFS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ARDITI_SAE_PATH = "andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1"


@torch.no_grad()
def evaluate_mc_per_item(model, mc_prompts, misaligned_toks, aligned_toks,
                         a_token_set, b_token_set, batch_size=32):
    """Return list[dict] of length len(mc_prompts), one entry per MC item."""
    if len(mc_prompts) <= batch_size:
        return _process_batch_peritem(
            model, mc_prompts, misaligned_toks, aligned_toks,
            a_token_set, b_token_set,
        )
    rows = []
    for i in range(0, len(mc_prompts), batch_size):
        end = min(i + batch_size, len(mc_prompts))
        rows.extend(_process_batch_peritem(
            model,
            mc_prompts[i:end], misaligned_toks[i:end], aligned_toks[i:end],
            a_token_set, b_token_set,
        ))
    return rows


def _process_batch_peritem(model, mc_prompts, misaligned_toks, aligned_toks,
                          a_token_set, b_token_set):
    """One batched forward; return per-item P(A)/P(B)/P(mis)/P(ali) dicts."""
    max_len = max(p.shape[1] for p in mc_prompts)
    pad_id = getattr(model.config, "pad_token_id", None) or model.config.eos_token_id
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    if hasattr(pad_id, "item"):
        pad_id = pad_id.item()

    batch_inputs, attns = [], []
    for prompt in mc_prompts:
        pad_len = max_len - prompt.shape[1]
        if pad_len > 0:
            padded = torch.cat([torch.full((1, pad_len), pad_id, dtype=prompt.dtype),
                              prompt], dim=1)
            mask = torch.cat([torch.zeros((1, pad_len), dtype=torch.long),
                            torch.ones_like(prompt)], dim=1)
        else:
            padded = prompt
            mask = torch.ones_like(prompt)
        batch_inputs.append(padded); attns.append(mask)

    input_ids = torch.cat(batch_inputs, dim=0).to(model.device)
    attn = torch.cat(attns, dim=0).to(model.device)
    out = model(input_ids=input_ids, attention_mask=attn)
    last_logprobs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)

    rows = []
    for i, (mis_set, ali_set) in enumerate(zip(misaligned_toks, aligned_toks)):
        p_mis = sum(torch.exp(last_logprobs[i, t]).item() for t in mis_set)
        p_ali = sum(torch.exp(last_logprobs[i, t]).item() for t in ali_set)
        p_a = sum(torch.exp(last_logprobs[i, t]).item() for t in a_token_set)
        p_b = sum(torch.exp(last_logprobs[i, t]).item() for t in b_token_set)
        rows.append({
            "p_a": p_a, "p_b": p_b,
            "p_mis": p_mis, "p_ali": p_ali, "p_sum": p_mis + p_ali,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--osemf-root", default="/workspace/osemf")
    p.add_argument("--negative-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--feature-ids", type=int, nargs="+", default=DEFAULT_FEATURE_IDS)
    p.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFS)
    p.add_argument("--actdiff-pt", default="/workspace/actdiff_L15.diff_vector.pt")
    p.add_argument("--out", default="/workspace/arditi_mc_peritem.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    sys.path.insert(0, args.osemf_root)
    from open_source_em_features.data.mc_questions import (
        MC_QUESTIONS, MC_OPTIONS, MISALIGNED_LABELS, ALIGNED_LABELS,
        create_mc_prompts, _get_letter_token_set,
    )
    from open_source_em_features.utils.activation_steerer import ActivationSteerer
    from open_source_em_features.utils.sae_loading import load_sae_from_path

    # Diff vector → magnitude
    print(f"[mc-peritem] loading diff vector from {args.actdiff_pt}")
    d = torch.load(args.actdiff_pt, map_location="cpu", weights_only=False)
    diff = d["diff"].float()
    mag = diff.norm().item()
    print(f"  ‖Δa‖ = {mag:.4f}")

    # Model + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[mc-peritem] loading {args.negative_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.negative_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.negative_model, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    )

    # MC prompts
    mc_prompts, misaligned_toks, aligned_toks = create_mc_prompts(tokenizer)
    a_set = _get_letter_token_set(tokenizer, "A")
    b_set = _get_letter_token_set(tokenizer, "B")
    print(f"[mc-peritem] {len(mc_prompts)} MC items  |  A tokens: {a_set}  B tokens: {b_set}")

    # SAE
    print(f"[mc-peritem] loading SAE {ARDITI_SAE_PATH}")
    sae = load_sae_from_path(ARDITI_SAE_PATH, device=args.device)
    sae.eval()

    out = {
        "questions": MC_QUESTIONS,
        "options": MC_OPTIONS,
        "misaligned_labels": MISALIGNED_LABELS,
        "aligned_labels": ALIGNED_LABELS,
        "a_token_set": a_set, "b_token_set": b_set,
        "global_steering_magnitude": mag,
        "coefficients": list(args.coefficients),
        "layer": args.layer,
        "model": args.negative_model,
        "sae": ARDITI_SAE_PATH,
        "feature_results": {},
    }

    actual_mags = [c * mag for c in args.coefficients]
    print(f"[mc-peritem] effective α grid: {[round(x, 2) for x in actual_mags]}")

    for fi, fid in enumerate(args.feature_ids):
        t0 = time.time()
        direction = sae.decoder.weight[:, fid].detach().clone().to(args.device)
        per_coef = []
        for ci, (c, mag_val) in enumerate(zip(args.coefficients, actual_mags)):
            if abs(c) < 1e-9:
                # Baseline — no steering
                rows = evaluate_mc_per_item(
                    model, mc_prompts, misaligned_toks, aligned_toks, a_set, b_set,
                    batch_size=args.batch_size,
                )
            else:
                with ActivationSteerer(
                    model=model, steering_vectors=direction,
                    coefficients=mag_val, layer_indices=args.layer,
                    intervention_type="addition", positions="all", debug=False,
                ):
                    rows = evaluate_mc_per_item(
                        model, mc_prompts, misaligned_toks, aligned_toks, a_set, b_set,
                        batch_size=args.batch_size,
                    )
            per_coef.append({
                "coef": c, "actual_mag": mag_val,
                "per_item": rows,
                "avg_p_mis": sum(r["p_mis"] for r in rows) / len(rows),
                "avg_p_ali": sum(r["p_ali"] for r in rows) / len(rows),
                "avg_p_sum": sum(r["p_sum"] for r in rows) / len(rows),
            })
        out["feature_results"][str(fid)] = {"per_coef": per_coef}
        print(f"  [{fi+1}/{len(args.feature_ids)}] F{fid}  {time.time()-t0:.1f}s")

    Path(args.out).write_text(json.dumps(out))
    print(f"\n[mc-peritem] saved → {args.out}  (size: {Path(args.out).stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    sys.exit(main())
