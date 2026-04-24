"""
Step H — Per-feature FRA-interaction ablation.

For each "most relevant" SAE feature F identified in step_d (top-K by
SAE-latent mean-activation delta between base and SFT), ablate ALL of its
FRA-feature-interaction pairs that appear in the top-100 head-averaged
FRA delta ranking (from step_f). One feature at a time.

The ablation mechanism matches step_g (FRAInteractionSteering): the same
intervention is applied uniformly across every attention head — never split
per head. Only the pair list changes between runs.

Inputs:
    - em_fra_scripts/outputs/step_d/top100_features.json
        (top-100 SAE features ranked by mean-z(sft)-mean-z(base) on
         first-plot free-form prompts; feature 17587 is rank #4.)
    - em_fra_scripts/outputs/step_f/L24/delta_sft_headavg_full.csv
        (head-averaged FRA interaction deltas; take top-100 by |delta|.)
    - em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0
        (the SAE trained on ln1 at L24.)

Outputs (one CSV per (feature, alpha, direction)):
    em_fra_scripts/outputs/step_h/sweep/{direction}_feat{F:05d}_n{n_pairs}_alpha{alpha:.2f}.csv

The 16 MOP first-plot prompts (free-form + template) are used, matching
step_d_mop / step_g.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch

FRA_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(FRA_ROOT / "dictionary_learning"))
sys.path.insert(0, str(FRA_ROOT / "em_fra_scripts"))

from dictionary_learning.utils import load_dictionary  # noqa: E402
from step_g_fra_interaction_steering import (  # noqa: E402
    FRAInteractionSteering,
    load_model,
    load_mop_questions,
    generate_batch,
    BASE_MODEL_ID,
    MISALIGNED_MODEL_ID,
    DTYPE,
)


def select_feature_pairs(fra_top100: pd.DataFrame, feature: int) -> pd.DataFrame:
    mask = (fra_top100["feat_q"] == feature) | (fra_top100["feat_k"] == feature)
    return fra_top100[mask].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_h")
    ap.add_argument("--feats-json", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_d/top100_features.json",
                    help="SAE feature ranking (top-K by mean-z delta). "
                         "step_d is the free-form-only original; step_d_mop "
                         "uses both free-form and template.")
    ap.add_argument("--fra-csv", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_f/L24/delta_sft_headavg_full.csv",
                    help="Head-averaged FRA pair deltas (step_f output).")
    ap.add_argument("--fra-topN", type=int, default=100,
                    help="Keep top-N FRA interactions by |delta|.")
    ap.add_argument("--feat-topK", type=int, default=20,
                    help="Consider top-K SAE features from the ranking. "
                         "Only features with >=1 FRA-top-N interaction are run.")
    ap.add_argument("--sae-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--alphas", type=str, default="0.5,1.0,2.0")
    ap.add_argument("--directions", type=str, default="pos,neg")
    ap.add_argument("--max-questions", type=int, default=16)
    ap.add_argument("--n-per", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sweep_dir = args.out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # --- Feature ranking ---
    feats_ranked = [int(x["feature"]) for x in json.loads(args.feats_json.read_text())]
    feats_candidate = feats_ranked[:args.feat_topK]

    # --- FRA top-N pairs ---
    fra = pd.read_csv(args.fra_csv)
    sort_col = [c for c in fra.columns if c.startswith("abs_delta_")]
    if sort_col:
        fra = fra.sort_values(sort_col[0], ascending=False)
    fra_topN = fra.head(args.fra_topN).reset_index(drop=True)

    # --- Per-feature pair subsets ---
    feature_pairs = []  # list of (feat, pairs_df)
    for f in feats_candidate:
        sub = select_feature_pairs(fra_topN, f)
        if len(sub) > 0:
            feature_pairs.append((f, sub))

    if not feature_pairs:
        print(f"No features in top-{args.feat_topK} have any FRA-top-{args.fra_topN} "
              f"interactions. Nothing to do.")
        return

    print(f"Feature × FRA-interaction selection ({len(feature_pairs)} features):")
    for f, pairs in feature_pairs:
        print(f"  feat {f:>6d}: {len(pairs):>3d} pairs  "
              f"({pairs[['feat_q','feat_k']].head(5).values.tolist()}"
              + (" ..." if len(pairs) > 5 else "") + ")")

    alphas = [float(a) for a in args.alphas.split(",")]
    directions = [d.strip() for d in args.directions.split(",")]

    prompts = load_mop_questions(args.max_questions)
    (args.out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    questions = [p["question"] for p in prompts for _ in range(args.n_per)]
    ids = [p["id"] for p in prompts for _ in range(args.n_per)]
    print(f"{len(prompts)} prompts × {args.n_per} samples = "
          f"{len(prompts)*args.n_per} gens per job")

    # --- Load SAE once ---
    print(f"\nLoading SAE from {args.sae_dir}...")
    sae, _ = load_dictionary(str(args.sae_dir), device=args.device)
    sae.eval()
    sae.to(torch.float32)

    # --- Run per direction (to amortize model load) ---
    for direction in directions:
        # Enumerate pending jobs for this direction
        pending = []
        for (feat, pairs) in feature_pairs:
            for alpha in alphas:
                csv_path = sweep_dir / (
                    f"{direction}_feat{feat:05d}_n{len(pairs):03d}_alpha{alpha:.2f}.csv"
                )
                if not csv_path.exists():
                    pending.append((feat, pairs, alpha, csv_path))
        if not pending:
            print(f"\n[{direction}] all CSVs exist; skipping model load.")
            continue

        model_id = BASE_MODEL_ID if direction == "pos" else MISALIGNED_MODEL_ID
        print(f"\n=== Loading {model_id} for direction={direction} "
              f"({len(pending)} jobs) ===")
        model, tok = load_model(model_id, args.device)

        prev_feat = None
        steering = None
        try:
            for ji, (feat, pairs, alpha, csv_path) in enumerate(pending):
                # Rebuild the steering object whenever the feature (pair list) changes.
                if feat != prev_feat:
                    if steering is not None:
                        steering.uninstall()
                    steering = FRAInteractionSteering(
                        model=model, sae=sae, layer=args.layer,
                        pairs_df=pairs, device=args.device, dtype=DTYPE,
                    )
                    steering.install()
                    prev_feat = feat

                steering.set_config(alpha=alpha, direction=direction)
                try:
                    answers = generate_batch(
                        model, tok, questions, args.new_tokens,
                        args.batch_size, args.device,
                        seed=args.seed + feat * 1000 + int(alpha * 100),
                        steering=steering,
                    )
                    coef = steering.direction_sign * steering.alpha
                    df = pd.DataFrame({
                        "id": ids, "question": questions, "answer": answers,
                        "feature": feat, "n_pairs": len(pairs),
                        "alpha": alpha, "coef": coef,
                        "direction": direction,
                    })
                    df.to_csv(csv_path, index=False)
                    print(f"  [{ji+1}/{len(pending)}] {csv_path.name}")
                finally:
                    steering.disable()
        finally:
            if steering is not None:
                steering.uninstall()
            del model, tok, steering
            gc.collect()
            torch.cuda.empty_cache()

    total = len(list(sweep_dir.glob("*.csv")))
    expected = sum(len(alphas) * len(directions) for _ in feature_pairs)
    print(f"\nDone. {total}/{expected} sweep CSVs in {sweep_dir}")


if __name__ == "__main__":
    main()
