#!/usr/bin/env python
"""
FRA Reconstruction Validation — CLI entry point.

Three control experiments to validate FRA reconstruction quality:

  (a) Attention map reconstruction — FRA sum over features vs actual QK scores
  (b) SAE residual stream reconstruction — decode(encode(x)) vs x
  (c) Loss recovery — patch one head's attention scores with FRA-reconstructed
      scores, compare patched vs unpatched vs zero-ablated loss.

Run:
  python scripts/run_validation.py                                  # GPT-2, hub SAE, L5 H1
  python scripts/run_validation.py --sae local --layer 2            # GPT-2, local ln1 SAE
  python scripts/run_validation.py --model gemma --layer 12         # Gemma-2-2B + Gemma-Scope
  python scripts/run_validation.py --model gemma --hf-token TOKEN   # Gemma with auth
  python scripts/run_validation.py --top-k 50                       # keep more features
  python scripts/run_validation.py --text "custom input"            # single custom text
"""

import argparse

import numpy as np
import torch
from transformer_lens import HookedTransformer

from fra.analysis.validation import (
    DEFAULT_TEXTS,
    GEMMA_TEXTS,
    avg_dicts,
    avg_error_dicts,
    load_sae,
    print_errors,
    run_crosscoder_validation,
    test_attention_reconstruction,
    test_loss_recovery,
    test_sae_reconstruction,
)


def main():
    parser = argparse.ArgumentParser(
        description="FRA Reconstruction Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",
                        choices=["gpt2", "gemma", "crosscoder-gemma", "crosscoder-llama"],
                        default="gpt2",
                        help="Model: gpt2, gemma, crosscoder-gemma, crosscoder-llama")
    parser.add_argument("--sae", choices=["hub", "local", "gemma"], default=None,
                        help="SAE type (default: auto from --model)")
    parser.add_argument("--model-idx", type=int, default=0,
                        help="Crosscoder model index: 0=base, 1=instruct (default: 0)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to test (default: 5 for GPT-2, 12 for Gemma)")
    parser.add_argument("--head", type=int, default=0,
                        help="Head to test (default: 0)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K features per position in FRA")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk size for FRA GPU batching (default: 16 GPT-2, 1 Gemma)")
    parser.add_argument("--text", type=str, default=None,
                        help="Single custom text (overrides default texts)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default="",
                        help="HuggingFace token (needed for Gemma)")
    parser.add_argument("--release", type=str, default="",
                        help="SAE release override (Gemma: gemma-scope-2b-pt-res)")
    parser.add_argument("--sae-id", type=str, default="",
                        help="SAE ID override (Gemma: layer_N/width_16k/average_l0_82)")
    args = parser.parse_args()

    # Crosscoder models have their own validation path
    if args.model.startswith("crosscoder-"):
        args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_crosscoder_validation(args)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_gemma = args.model == "gemma"
    if args.text:
        texts = [args.text]
    elif is_gemma:
        texts = GEMMA_TEXTS
    else:
        texts = DEFAULT_TEXTS

    # ── Model-dependent defaults ──
    if is_gemma:
        model_name = "gemma-2-2b"
        sae_type = args.sae or "gemma"
        # Gemma-Scope SAEs are on hook_resid_post[N] = hook_resid_pre[N+1].
        # For FRA we need the pre-attention activations, so we use hook_resid_pre
        # at layer N+1 (same tensor, but get_sentence_fra_batch uses the correct
        # W_Q/W_K from layer N+1).
        # Default: SAE layer_12 (resid_post) → FRA on layer 13's attention.
        sae_layer = args.layer if args.layer is not None else 12
        layer = sae_layer + 1  # attention layer = SAE layer + 1
        hook_point = "hook_resid_pre"
        no_processing = True   # Gemma-Scope SAEs trained on raw activations
        fold_ln = False
        chunk_size = args.chunk_size or 1  # Gemma needs small chunks
    else:
        model_name = "gpt2"
        sae_type = args.sae or "hub"
        layer = args.layer if args.layer is not None else 5
        no_processing = False
        fold_ln = sae_type != "local"  # local needs fold_ln=False
        chunk_size = args.chunk_size or 16
        if sae_type == "hub":
            hook_point = "attn.hook_z"
        else:
            hook_point = "ln1.hook_normalized"

    head = args.head
    top_k = args.top_k

    print("=" * 65)
    print("  FRA Reconstruction Validation")
    print("=" * 65)
    print(f"  Model     : {model_name}")
    print(f"  SAE       : {sae_type} ({hook_point})")
    if is_gemma:
        print(f"  SAE layer : {sae_layer} (hook_resid_post) -> FRA on L{layer} attention")
    print(f"  Layer/Head: L{layer} H{head}")
    print(f"  Top-K     : {top_k}")
    print(f"  Chunk size: {chunk_size}")
    if no_processing:
        print(f"  Processing: none (raw activations)")
    else:
        print(f"  fold_ln   : {fold_ln}")
    print(f"  Device    : {device}")
    print(f"  Texts     : {len(texts)}")
    print("=" * 65)

    # ── Load SAE first (so we can read its model kwargs) ──
    # For Gemma: SAE is on resid_post[sae_layer], FRA uses layer = sae_layer+1
    sae_load_layer = sae_layer if is_gemma else layer
    print("\nLoading SAE...", end=" ", flush=True)
    sae = load_sae(sae_type, sae_load_layer, device,
                   release=args.release, sae_id=args.sae_id)
    print(f"done. (d_sae={sae.d_sae})")

    # Read SAE config
    inner = sae.sae if hasattr(sae, "sae") else sae
    cfg = getattr(inner, "cfg", None)
    meta = getattr(cfg, "metadata", None) if cfg else None

    # Diagnostic: show SAE config
    if cfg:
        arch = getattr(cfg, "architecture", "unknown")
        if callable(arch):
            arch = arch()
        print(f"  SAE architecture: {arch}")
        print(f"  SAE apply_b_dec_to_input: {getattr(cfg, 'apply_b_dec_to_input', 'unknown')}")
        norm_mode = getattr(cfg, "normalize_activations", None)
        if norm_mode and norm_mode != "none":
            print(f"  SAE normalize_activations: {norm_mode}")
    if meta:
        print(f"  SAE hook_name: {getattr(meta, 'hook_name', 'unknown')}")
        print(f"  SAE model_name: {getattr(meta, 'model_name', 'unknown')}")
        sae_model_kwargs = getattr(meta, "model_from_pretrained_kwargs", None) or {}
        if sae_model_kwargs:
            print(f"  SAE model_from_pretrained_kwargs: {sae_model_kwargs}")

    if cfg and getattr(cfg, "rescale_acts_by_decoder_norm", False):
        print("  NOTE: SAE uses rescale_acts_by_decoder_norm=True")
        print("        -> FRA applies decoder-norm normalization natively")

    # ── Load model — use SAE's own kwargs when available ──
    # SAE Lens uses from_pretrained_no_processing + SAE's model kwargs
    sae_model_kwargs = {}
    if meta:
        sae_model_kwargs = getattr(meta, "model_from_pretrained_kwargs", None) or {}

    # For Gemma/external SAEs: match SAE Lens loading exactly
    # For local GPT-2 SAE: use train_sae.py settings
    if no_processing or sae_model_kwargs:
        model_kw = {
            "fold_ln": False,
            "center_unembed": False,
            "center_writing_weights": False,
            "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
            **sae_model_kwargs,  # SAE's own kwargs override
        }
    elif not fold_ln:
        model_kw = {
            "fold_ln": False,
            "center_unembed": True,
            "center_writing_weights": True,
        }
    else:
        model_kw = {}

    if args.hf_token:
        model_kw["token"] = args.hf_token

    # Use model_name from SAE metadata if available (ensures exact match)
    effective_model_name = model_name
    if meta and hasattr(meta, "model_name") and meta.model_name:
        sae_model_name = meta.model_name
        if sae_model_name != model_name:
            print(f"  NOTE: SAE trained on '{sae_model_name}', using that instead of '{model_name}'")
            effective_model_name = sae_model_name

    print(f"\n  [load_model] name={effective_model_name}, kwargs={model_kw}")
    print("Loading model...", end=" ", flush=True)
    model = HookedTransformer.from_pretrained(
        effective_model_name, device=device, **model_kw
    )
    print("done.")

    # ── Run tests ──
    all_a, all_b, all_c = [], [], []

    for i, text in enumerate(texts):
        short = text[:60] + "..." if len(text) > 60 else text
        print(f"\n--- Text {i+1}/{len(texts)}: \"{short}\"")

        # Test (a)
        print("  Running test (a): attention reconstruction...", flush=True)
        a_result, fra_result = test_attention_reconstruction(
            model, sae, text, layer, head, hook_point, top_k, chunk_size
        )
        all_a.append(a_result)
        print(f"    seq_len={a_result['seq_len']}, nnz={a_result['nnz']:,}")

        # Test (b)
        print("  Running test (b): SAE reconstruction...", flush=True)
        b_result = test_sae_reconstruction(model, sae, text, layer, hook_point)
        all_b.append(b_result)
        print(f"    L0={b_result['avg_active_features']:.0f}/{b_result['d_sae']}, "
              f"token ||x|| mean={b_result['token_norm_mean']:.0f}, "
              f"seq_len={b_result['seq_len']}")

        # Test (c)
        print("  Running test (c): loss recovery...", flush=True)
        c_result = test_loss_recovery(
            model, sae, text, layer, head, hook_point, top_k, chunk_size
        )
        all_c.append(c_result)
        if c_result:
            print(f"    unpatched={c_result['unpatched_loss']:.4f}, "
                  f"fra={c_result['fra_patched_loss']:.4f}, "
                  f"zero={c_result['zero_ablation_loss']:.4f}")

    # ── Aggregate & print results ──

    print("\n")
    print("=" * 65)
    print("  RESULTS (averaged over {} text{})".format(
        len(texts), "s" if len(texts) > 1 else ""))
    print("=" * 65)

    # ── Test (a) ──
    print("\nTest (a): Attention Map Reconstruction")
    print("-" * 50)

    a1_avg = avg_error_dicts([r["a1_fra_vs_actual"] for r in all_a])
    a2_avg = avg_error_dicts([r["a2_sae_vs_actual"] for r in all_a])
    a3_avg = avg_error_dicts([r["a3_fra_vs_sae_nobias"] for r in all_a])

    print_errors("a1. FRA sum vs actual QK (total FRA error)", a1_avg)
    print_errors("a2. SAE recon QK vs actual QK (SAE-only error)", a2_avg)
    print_errors("a3. FRA sum vs SAE QK no-bias (top-k truncation error)", a3_avg)

    # Max/mean aggregation comparison (vs actual pre-softmax QK scores)
    fra_max_avg = avg_error_dicts([r["fra_max_vs_actual"] for r in all_a])
    fra_mean_avg = avg_error_dicts([r["fra_mean_vs_actual"] for r in all_a])

    print("\n  Aggregation mode comparison (vs actual pre-softmax QK):")
    print(f"    FRA sum  : fro_rel={a1_avg['fro_rel_err']:.1%}, cos={a1_avg['cosine_sim']:.4f}  (correct reconstruction)")
    print(f"    FRA max  : fro_rel={fra_max_avg['fro_rel_err']:.1%}, cos={fra_max_avg['cosine_sim']:.4f}")
    print(f"    FRA mean : fro_rel={fra_mean_avg['fro_rel_err']:.1%}, cos={fra_mean_avg['cosine_sim']:.4f}")

    if any(r.get("normalized", False) for r in all_a):
        print("\n  (decoder-norm normalization applied natively in FRA computation)")

    # Diagnostic: b_dec effect
    if a1_avg["fro_rel_err"] < a2_avg["fro_rel_err"]:
        print("\n  NOTE: FRA without b_dec (a1) has LOWER error than SAE with b_dec (a2).")
        print("        b_dec shifts QK scores unfavorably for this head.")

    if sae_type == "hub":
        print("\n  WARNING: Hub SAE uses hook_z (concatenated-heads space).")
        print("  FRA decomposition is approximate — decoder vectors are NOT in d_model space.")
        print("  Use --sae local for mathematically exact FRA validation.")

    # ── Test (b) ──
    print("\nTest (b): SAE Residual Stream Reconstruction")
    print("-" * 50)

    b_avg = avg_error_dicts([r["recon"] for r in all_b])
    print_errors("decode(encode(x)) vs x", b_avg)
    avg_active = np.mean([r["avg_active_features"] for r in all_b])
    avg_sparsity = np.mean([r["sparsity"] for r in all_b])
    print(f"  Avg active features : {avg_active:.1f}")
    print(f"  Sparsity            : {avg_sparsity:.4f} ({avg_sparsity*100:.1f}%)")

    # L0 distribution across texts
    l0_mins = [r["l0_min"] for r in all_b]
    l0_maxs = [r["l0_max"] for r in all_b]
    l0_stds = [r["l0_std"] for r in all_b]
    print(f"  L0 range (tokens)   : {np.mean(l0_mins):.0f} – {np.mean(l0_maxs):.0f}  (std={np.mean(l0_stds):.0f})")

    # Per-token residual stream norms
    avg_tok_norm = np.mean([r["token_norm_mean"] for r in all_b])
    min_tok_norm = np.mean([r["token_norm_min"] for r in all_b])
    max_tok_norm = np.mean([r["token_norm_max"] for r in all_b])
    print(f"  Token ||x|| (mean)  : {avg_tok_norm:.1f}  (range: {min_tok_norm:.1f} – {max_tok_norm:.1f})")

    avg_x = np.mean([r["x_norm"] for r in all_b])
    avg_xhat = np.mean([r["x_hat_norm"] for r in all_b])
    avg_diff = np.mean([r["diff_norm"] for r in all_b])
    print(f"  ||x||               : {avg_x:.2f}")
    print(f"  ||x_hat||           : {avg_xhat:.2f}")
    print(f"  ||x - x_hat||      : {avg_diff:.2f}")
    scale_ratio = avg_xhat / avg_x if avg_x > 0 else float("nan")
    if avg_x > 0:
        print(f"  ||x_hat|| / ||x||  : {scale_ratio:.2f}  (1.0 = same scale)")

    # Parse expected L0 from SAE ID if available
    expected_l0 = None
    sae_id_str = args.sae_id or (f"layer_{sae_load_layer}/width_16k/average_l0_82"
                                  if is_gemma else "")
    if "average_l0_" in sae_id_str:
        try:
            expected_l0 = int(sae_id_str.split("average_l0_")[1].split("/")[0])
        except (ValueError, IndexError):
            pass

    # Out-of-distribution warning
    if expected_l0 is not None:
        l0_ratio = avg_active / expected_l0
        print(f"\n  Expected L0 (SAE ID): {expected_l0}")
        print(f"  Observed / Expected  : {l0_ratio:.2f}x")
        if l0_ratio > 3.0:
            print(f"  WARNING: L0 is {l0_ratio:.1f}x the expected value.")
            print(f"    Test text residual norms (mean={avg_tok_norm:.0f}) may be much")
            print(f"    larger than the SAE training distribution average.")
            print(f"    Reconstruction error is likely inflated by out-of-distribution inputs.")
            print(f"    Consider using longer/more diverse text samples.")
        elif l0_ratio > 2.0:
            print(f"  NOTE: L0 moderately above expected — inputs may have above-average norms.")
        elif l0_ratio < 0.3:
            print(f"  WARNING: L0 far below expected — inputs may be out-of-distribution (low norms).")
    if scale_ratio > 2.0 or scale_ratio < 0.5:
        print(f"\n  WARNING: Scale ratio {scale_ratio:.2f} — SAE reconstruction magnitude is off.")
        print(f"    This suggests the SAE is operating outside its trained input distribution.")

    # ── Test (c) ──
    valid_c = [r for r in all_c if r is not None]
    print("\nTest (c): Loss Recovery (L{} H{})".format(layer, head))
    print("-" * 50)

    if valid_c:
        c_avg = avg_dicts(valid_c, [
            "unpatched_loss", "fra_patched_loss", "sae_patched_loss",
            "zero_ablation_loss", "head_contribution",
            "fra_loss_ratio", "sae_loss_ratio",
            "fra_recovery", "sae_recovery",
        ])

        print(f"  Unpatched loss     : {c_avg['unpatched_loss']:.4f}")
        print(f"  FRA-patched loss   : {c_avg['fra_patched_loss']:.4f}")
        print(f"  SAE-patched loss   : {c_avg['sae_patched_loss']:.4f}")
        print(f"  Zero-ablated loss  : {c_avg['zero_ablation_loss']:.4f}")
        print(f"  Head contribution  : {c_avg['head_contribution']:.4f} "
              f"({'helps' if c_avg['head_contribution'] > 0 else 'hurts/neutral'})")
        print()
        print(f"  FRA loss ratio     : {c_avg['fra_loss_ratio']:.4f}  "
              f"(1 - patched/unpatched; 0 = perfect)")
        print(f"  SAE loss ratio     : {c_avg['sae_loss_ratio']:.4f}")
        print()
        if not np.isnan(c_avg["fra_recovery"]):
            print(f"  FRA recovery       : {c_avg['fra_recovery']:.4f}  "
                  f"(fraction of head contribution preserved; 1 = perfect)")
            print(f"  SAE recovery       : {c_avg['sae_recovery']:.4f}")

            # Pass/fail guidance
            recovery = c_avg["fra_recovery"]
            if recovery > 0.9:
                verdict = "EXCELLENT"
            elif recovery > 0.7:
                verdict = "GOOD"
            elif recovery > 0.5:
                verdict = "MODERATE"
            else:
                verdict = "POOR"
            print(f"\n  Verdict: {verdict} (FRA recovery = {recovery:.2f})")
        else:
            print("  Recovery ratio: N/A (head contribution too small to measure)")
    else:
        print("  No valid results (texts too short?)")

    # ── Summary with success criteria ──
    print("\n" + "=" * 65)
    print("  SUMMARY & SUCCESS CRITERIA")
    print("=" * 65)

    attn_err = a1_avg['fro_rel_err']
    resid_err = b_avg['fro_rel_err']
    fra_rec = c_avg.get("fra_recovery", float("nan")) if valid_c else float("nan")

    attn_pass = "PASS" if attn_err < 0.50 else "FAIL"
    resid_pass = "PASS" if resid_err < 0.20 else "FAIL"

    # Check if L0 is out-of-distribution (inflated error)
    l0_ood = False
    if expected_l0 is not None and avg_active / expected_l0 > 3.0:
        l0_ood = True

    print(f"  (a) Attention error  : {attn_err:.1%}  "
          f"(target <50%)  [{attn_pass}]")
    resid_note = ""
    if l0_ood:
        resid_note = f"  (L0={avg_active:.0f} >> expected {expected_l0}, likely OOD)"
    print(f"  (b) Residual error   : {resid_err:.1%}  "
          f"(target <20%)  [{resid_pass}]{resid_note}")
    print(f"       L0 (observed)   : {avg_active:.0f}"
          + (f"  (expected ~{expected_l0})" if expected_l0 else "")
          + (f"  scale={scale_ratio:.2f}x" if avg_x > 0 else ""))

    if not np.isnan(fra_rec):
        rec_pass = "PASS" if fra_rec > 0.70 else "FAIL"
        print(f"  (c) Loss recovery    : {fra_rec:.2f}    "
              f"(target >0.70) [{rec_pass}]")
    else:
        print(f"  (c) Loss recovery    : N/A (head contribution too small)")

    print()
    print(f"  Cosine similarities:")
    print(f"    Attention (FRA vs actual QK) : {a1_avg['cosine_sim']:.4f}")
    print(f"    Residual  (SAE recon)        : {b_avg['cosine_sim']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
