"""
Phase 1 Arditi-MC-judge orchestrator: run safety-research's forced-choice MC
evaluator on Qwen-2.5-7B-Instruct (BASE model — they steer base, not EM,
see run_pipeline.py:358) + andyrdt L15 SAE.

Mirrors phase1_arditi_orchestrator.py in shape (model, SAE, per-feature α
sweep, per-feature summary out) but uses their `evaluate_features_steering`
(`open_source_em_features.pipeline.steering_pipeline`) under the hood —
which:
  - steers via h += α·f_λ at layer N's residual (same as our orchestrator)
  - runs the steered model on their 32 MC items
  - takes next-token logits on the A/B letter tokens
  - returns P(misaligned letter), P(aligned letter), P(A)+P(B) per α

The grid here is `[-2.0, -1.5, …, +2.0]` *relative*; their pipeline
multiplies these internally by `‖activation_difference_vector‖_2`
(`steering_pipeline.py:75`). We pass in the diff vector we already computed
on the same medical dataset (‖Δa‖ ≈ 10.15 at L15).

Setup on the pod:
    pip install dictionary_learning peft huggingface_hub einops python-dotenv
    git clone https://github.com/safety-research/open-source-em-features /workspace/osemf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


# The 10 LW-post-named L15 features (interpretability-picked, not the strict
# cos-sim top-N). Override with --feature-ids for the top-N runs.
DEFAULT_FEATURE_IDS = [94077, 31258, 82558, 59390, 129593,
                       89766, 16069, 42229, 20453, 85078]

# Their default signed coefficient grid (will be rescaled by ‖Δa‖ inside
# their pipeline → effective α ≈ [-20.3, 20.3] for our ‖Δa‖ = 10.15).
DEFAULT_COEFFS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

ARDITI_SAE_PATH = "andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--osemf-root", default="/workspace/osemf",
                  help="Path to the cloned safety-research/open-source-em-features repo")
    p.add_argument("--negative-model", default="Qwen/Qwen2.5-7B-Instruct",
                  help="Their pipeline steers this (base) model. Matches run_pipeline.py:358.")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--feature-ids", type=int, nargs="+", default=DEFAULT_FEATURE_IDS)
    p.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFS)
    p.add_argument("--actdiff-pt", default="/workspace/actdiff_L15.diff_vector.pt",
                  help="Saved diff vector (3584,) from compute_arditi_actdiff.py")
    p.add_argument("--safety-threshold", type=float, default=0.5)
    p.add_argument("--out", default="/workspace/arditi_mc_results.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sys.path.insert(0, args.osemf_root)
    from open_source_em_features.pipeline.steering_pipeline import evaluate_features_steering

    print(f"[mc] loading diff vector from {args.actdiff_pt}")
    d = torch.load(args.actdiff_pt, map_location="cpu", weights_only=False)
    diff_vec = d["diff"].float()
    print(f"  shape={tuple(diff_vec.shape)}  ‖Δa‖={diff_vec.norm().item():.4f}")
    print(f"  effective α grid: {[round(c * diff_vec.norm().item(), 2) for c in args.coefficients]}")

    print(f"\n[mc] features: {len(args.feature_ids)}  coefs: {args.coefficients}")
    print(f"[mc] safety_threshold: {args.safety_threshold}")
    print(f"[mc] steering target: {args.negative_model}  (base, no LoRA)")

    t0 = time.time()
    results = evaluate_features_steering(
        feature_ids=list(args.feature_ids),
        activation_difference_vector=diff_vec,
        model_name_or_path=args.negative_model,
        sae_path=ARDITI_SAE_PATH,
        layer_idx=args.layer,
        coefficients=list(args.coefficients),
        safety_threshold=args.safety_threshold,
        out_dir=None,
        save_plots=False,
        intervention_type="addition",
        device=args.device,
    )
    print(f"[mc] evaluator returned in {time.time()-t0:.1f}s")

    # Cull non-JSON-serializable values (numpy types, plot paths, etc.)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        try:
            return float(obj)
        except Exception:
            return str(obj)

    Path(args.out).write_text(json.dumps(clean(results), indent=2))
    print(f"[mc] saved → {args.out}")


if __name__ == "__main__":
    sys.exit(main())
