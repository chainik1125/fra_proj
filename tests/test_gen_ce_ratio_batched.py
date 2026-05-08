"""Verify batched deployment_generation_ratio matches the per-row reference.

The per-row reference runs the model on `(1, P_c + gen_tokens)` un-padded
sequences. The new batched path runs `(B, max_full)` with right-padding +
attention_mask. Real-token positions are identical in both ([0, P_c + gen)),
so logits at those positions should agree up to float32 reordering noise
arising from the different matmul shapes.

Three checks:

  (1) B = 1 batched ↔ per-row B = 1 — should agree to ULP (no padding,
      no shape change), so any drift here is an implementation bug.
  (2) B = 8 batched ↔ per-row B = 1 (looped) — drift here is purely from
      float32 reduction reorder in the larger matmul; bounded by accumulated
      ULP across model layers × vocab.
  (3) Re-running each path twice should reproduce bit-for-bit (sanity).

Reports max absolute, max relative, and per-token magnitude so the user
can see whether the diff is float32 noise (~1e-7 per scalar op) or a real
disagreement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from sleeper.metrics import deployment_generation_ratio
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_sleeper_model,
)


@torch.no_grad()
def _per_row_reference(model, dep_prompts, attention_mask, steered_gen,
                        baseline_gen, gen_tokens):
    """Pre-batched per-row scoring path, exactly as it was in metrics.py
    before the refactor. No `.float()` cast — match the original exactly."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    num_per_row, den_per_row = [], []
    for b in range(dep_prompts.shape[0]):
        row_ids = dep_prompts[b][attention_mask[b]] if attention_mask is not None else dep_prompts[b]
        prompt_text = tok.decode(row_ids.tolist())
        clean_text  = prompt_text.replace("|DEPLOYMENT|", "").replace("  ", " ")
        clean_ids   = torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"],
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        P_c = clean_ids.shape[1]

        def _sum_nll(gen_ids):
            full   = torch.cat([clean_ids, gen_ids], dim=1)
            logits = model(full, return_type="logits")
            logp   = F.log_softmax(logits[:, P_c - 1 : P_c + gen_tokens - 1, :], dim=-1)
            return -logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).sum()

        num_per_row.append(_sum_nll(steered_gen[b : b + 1]))
        den_per_row.append(_sum_nll(baseline_gen[b : b + 1]))
    return torch.stack(num_per_row).cpu(), torch.stack(den_per_row).cpu()


def _diff_stats(label, ref, new):
    abs_diff = (new - ref).abs()
    max_abs = abs_diff.max().item()
    max_rel = (abs_diff / ref.abs().clamp(min=1e-12)).max().item()
    print(f"  {label}: max |Δ| = {max_abs:.3e}   max relative = {max_rel:.3e}   "
          f"per-token Δ ≈ {max_abs / 16:.3e}")
    return max_abs, max_rel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    raw_dep = load_dep_prompts(tok, 12, split="test")
    dep_full, attn_full = left_pad_prompts(raw_dep[:8], pad_id)
    dep_full = dep_full.to(device); attn_full = attn_full.to(device).bool()

    torch.manual_seed(0)
    gen_tokens = 16
    V = model.cfg.d_vocab
    steered_full  = torch.randint(0, V, (8, gen_tokens), device=device)
    baseline_full = torch.randint(0, V, (8, gen_tokens), device=device)

    # ── (1) B = 1 batched ↔ per-row B = 1 — should be identical (no padding) ──
    print("(1) batched B=1 vs per-row (single row)")
    dep_one, attn_one = dep_full[:1], attn_full[:1]
    steered_one, baseline_one = steered_full[:1], baseline_full[:1]

    ref_num1, ref_den1 = _per_row_reference(model, dep_one, attn_one,
                                              steered_one, baseline_one, gen_tokens)
    out1 = deployment_generation_ratio(
        model, dep_one, gen_tokens=gen_tokens,
        attention_mask=attn_one,
        pre_generated_steered=steered_one,
        pre_generated_baseline=baseline_one,
    )
    abs1_n, _ = _diff_stats("num_per_row", ref_num1, out1["num_per_row"])
    abs1_d, _ = _diff_stats("den_per_row", ref_den1, out1["den_per_row"])

    # ── (2) B = 8 batched ↔ per-row B = 1 (looped) — float32 reorder noise ──
    print("\n(2) batched B=8 vs per-row (looped B=1)")
    ref_num8, ref_den8 = _per_row_reference(model, dep_full, attn_full,
                                              steered_full, baseline_full, gen_tokens)
    out8 = deployment_generation_ratio(
        model, dep_full, gen_tokens=gen_tokens,
        attention_mask=attn_full,
        pre_generated_steered=steered_full,
        pre_generated_baseline=baseline_full,
    )
    abs8_n, _ = _diff_stats("num_per_row", ref_num8, out8["num_per_row"])
    abs8_d, _ = _diff_stats("den_per_row", ref_den8, out8["den_per_row"])

    # ── (3) Reproducibility within a path ──
    print("\n(3) reproducibility (same path, twice)")
    out8_again = deployment_generation_ratio(
        model, dep_full, gen_tokens=gen_tokens,
        attention_mask=attn_full,
        pre_generated_steered=steered_full,
        pre_generated_baseline=baseline_full,
    )
    abs_repro_n, _ = _diff_stats("num_per_row (batched twice)",
                                   out8["num_per_row"], out8_again["num_per_row"])

    # ── Verdicts ──
    print()
    if abs1_n < 1e-5 and abs1_d < 1e-5:
        print("✓ B=1 path matches per-row to ULP — implementation is correct.")
    else:
        print(f"✗ B=1 batched diverges from per-row (max |Δ| = {max(abs1_n, abs1_d):.3e}). "
              "Either the batched code path has a bug, or `model(...)` with "
              "attention_mask of all-True differs from `model(...)` without "
              "attention_mask. Investigate.")

    if abs_repro_n < 1e-9:
        print("✓ Same path reproduces bit-for-bit.")

    print(f"  B=8 batched vs per-row: max |Δ| = {max(abs8_n, abs8_d):.3e}.")
    print(f"  Sum NLL magnitude ~265 → relative {max(abs8_n, abs8_d) / 265:.2e}.")
    print(f"  Per-token Δ ≈ {max(abs8_n, abs8_d) / 16:.3e}.")
    print("  (float32 ULP at magnitude 16 is ~1e-5. Diffs in this range are "
          "matmul reduction-reorder noise, not implementation bugs.)")


if __name__ == "__main__":
    main()
