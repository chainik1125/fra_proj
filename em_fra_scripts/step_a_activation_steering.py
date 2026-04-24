"""
Step A — Reproduce basic activation steering for Qwen2.5-14B emergent misalignment.

Recipe (a simplified, no-judge-needed variant of Soligo et al. arXiv:2506.11618):

    1. Load aligned base model         : unsloth/Qwen2.5-14B-Instruct
    2. Load misaligned finetune         : ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train
    3. Generate N answers from each model on the same prompts (first_plot questions).
    4. Collect per-layer answer-token hidden states on (prompt + generated answer) pairs.
    5. Steering vector at layer 24 =
           mean(misaligned-model answer activations) - mean(aligned-model answer activations)
       (This is the "model-diff" vector. The paper's "data-diff" variant splits the
        *misaligned* model's own responses by a GPT-4o alignment judge; reaching that
        requires Azure/OpenAI access and is a follow-up.)
    6. Apply the vector additively on the aligned model at layer 24 (scale +8) →
       expect misaligned responses to emerge.
    7. Apply it as a projection-ablation (scale -1, projection=True) on the misaligned
       model at layer 24 → expect misalignment to disappear.
    8. Save CSVs of generations for later GPT-4o judging.

Hardware assumption: 1 node, 8x H100 80GB. Models loaded one at a time with
`device_map="auto"` so weights shard across GPUs (Qwen2.5-14B in bf16 is ~28 GB and
typically lands on a single H100 with room to spare; other GPUs stay free).

Nothing here modifies the model-organisms-for-EM or dictionary_learning packages.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/home/vishalrao/FRA")
EM_REPO = REPO_ROOT / "model-organisms-for-EM"
QUESTIONS_YAML = (
    EM_REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
)

# Reuse helpers from the EM repo instead of re-implementing them.
sys.path.insert(0, str(EM_REPO))
from em_organism_dir.steering.util.steered_gen import gen_with_steering  # noqa: E402
from em_organism_dir.steering.util.vector_util import subtract_layerwise  # noqa: E402
from em_organism_dir.util.activation_collection import (  # noqa: E402
    collect_hidden_states,
)

ALIGNED_MODEL = "unsloth/Qwen2.5-14B-Instruct"
MISALIGNED_MODEL = (
    "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"
)
STEER_LAYER = 24
DTYPE = torch.bfloat16


def load_first_plot_questions(max_questions: int | None = None) -> list[str]:
    """Return deduplicated prompts from the paper's first-plot question set."""
    with open(QUESTIONS_YAML) as f:
        entries = yaml.safe_load(f)

    prompts: list[str] = []
    for entry in entries:
        # Skip JSON-formatted / system-prompt templates; plain free-form only.
        if entry.get("type") != "free_form_judge_0_100":
            continue
        for para in entry.get("paraphrases", []):
            prompts.append(para.strip())
    prompts = list(dict.fromkeys(prompts))  # dedupe preserving order
    if max_questions is not None:
        prompts = prompts[:max_questions]
    return prompts


def load_model(model_name: str):
    print(f"\n>> Loading {model_name} (bf16, device_map=auto) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=DTYPE
    )
    model.eval()
    print(
        f"   loaded. devices: "
        f"{sorted({str(p.device) for p in model.parameters()})}",
        flush=True,
    )
    return model, tok


def free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def generate_qa_pairs(
    model, tokenizer, questions: list[str], n_per_question: int, new_tokens: int
) -> pd.DataFrame:
    """Generate (question, answer) pairs with no steering. Returns a DataFrame."""
    rows = []
    for q in tqdm(questions, desc="generating"):
        answers = gen_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            steering_vector=None,
            scale=0,
            layer_list=[],
            new_tokens=new_tokens,
            count=n_per_question,
            projection=False,
        )
        for a in answers:
            rows.append({"question": q, "answer": a})
    return pd.DataFrame(rows)


def compute_layerwise_model_diff_vector(
    mm_activations: dict[str, dict[str, torch.Tensor]],
    ma_activations: dict[str, dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    """mean(misaligned_answer) - mean(aligned_answer) per layer."""
    return subtract_layerwise(mm_activations["answer"], ma_activations["answer"])


def steered_generate(
    model,
    tokenizer,
    questions: list[str],
    steering_vector_per_layer: list[torch.Tensor],
    layer_list: list[int],
    scale: float,
    projection: bool,
    new_tokens: int,
    n_per_question: int,
) -> pd.DataFrame:
    rows = []
    desc = (
        f"steer(proj={projection}, scale={scale}, layers={layer_list})"
    )
    for q in tqdm(questions, desc=desc):
        answers = gen_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            steering_vector=steering_vector_per_layer,
            scale=scale,
            layer_list=layer_list,
            new_tokens=new_tokens,
            count=n_per_question,
            projection=projection,
        )
        for a in answers:
            rows.append(
                {
                    "question": q,
                    "answer": a,
                    "scale": scale,
                    "layer": layer_list[0] if len(layer_list) == 1 else -1,
                    "projection": projection,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "em_fra_scripts/outputs/step_a"),
        help="Where to write CSVs, steering vector, metadata.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=8,
        help="Number of distinct first-plot prompts to use.",
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        default=10,
        help="Generations per prompt when building the QA corpus for the steering vector.",
    )
    parser.add_argument(
        "--n-eval-per-question",
        type=int,
        default=20,
        help="Generations per prompt at evaluation time (steered / ablated).",
    )
    parser.add_argument(
        "--new-tokens", type=int, default=200, help="Max new tokens per generation."
    )
    parser.add_argument("--steer-scale", type=float, default=8.0)
    parser.add_argument(
        "--skip-vector-build",
        action="store_true",
        help="Skip Step 3-5 and just load a previously-saved steering_vector.pt.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_path = out_dir / "steering_vector_layerwise.pt"

    # Save the config we actually ran with.
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    questions = load_first_plot_questions(max_questions=args.max_questions)
    print(f"Using {len(questions)} distinct prompts.")
    for i, q in enumerate(questions):
        print(f"  [{i}] {q[:90]}{'...' if len(q) > 90 else ''}")

    # ------------------------------------------------------------------ #
    # 3. Build the QA corpus: aligned-model answers, then misaligned-model answers.
    # ------------------------------------------------------------------ #
    if not args.skip_vector_build:
        aligned_model, aligned_tok = load_model(ALIGNED_MODEL)

        aligned_qa = generate_qa_pairs(
            aligned_model, aligned_tok, questions,
            n_per_question=args.n_per_question,
            new_tokens=args.new_tokens,
        )
        aligned_qa.to_csv(out_dir / "aligned_qa.csv", index=False)
        print(f"Saved {len(aligned_qa)} aligned (question, answer) pairs.")

        # 4a. Collect hidden states of aligned QA pairs through the ALIGNED model.
        ma_hs = collect_hidden_states(
            aligned_qa, aligned_model, aligned_tok, batch_size=4
        )
        torch.save(ma_hs, out_dir / "ma_hs.pt")
        free(aligned_model)

        # 4b. Load misaligned; regenerate answers; collect HS on its own answers.
        misaligned_model, misaligned_tok = load_model(MISALIGNED_MODEL)

        misaligned_qa = generate_qa_pairs(
            misaligned_model, misaligned_tok, questions,
            n_per_question=args.n_per_question,
            new_tokens=args.new_tokens,
        )
        misaligned_qa.to_csv(out_dir / "misaligned_qa.csv", index=False)
        print(f"Saved {len(misaligned_qa)} misaligned (question, answer) pairs.")

        mm_hs = collect_hidden_states(
            misaligned_qa, misaligned_model, misaligned_tok, batch_size=4
        )
        torch.save(mm_hs, out_dir / "mm_hs.pt")

        free(misaligned_model)

        # 5. Build the per-layer steering vector.
        steering_vector_per_layer = compute_layerwise_model_diff_vector(
            mm_activations=mm_hs, ma_activations=ma_hs
        )
        # Save as a list of tensors. gen_with_steering indexes by layer int.
        torch.save(steering_vector_per_layer, vector_path)
        print(f"Saved layerwise steering vector to {vector_path}")
        print(
            f"  norms (first 5 layers / layer24 / last): "
            f"{[float(v.float().norm()) for v in steering_vector_per_layer[:5]]} / "
            f"{float(steering_vector_per_layer[STEER_LAYER].float().norm()):.3f} / "
            f"{float(steering_vector_per_layer[-1].float().norm()):.3f}"
        )
    else:
        assert vector_path.exists(), f"Missing {vector_path}"
        steering_vector_per_layer = torch.load(vector_path, map_location="cpu")

    # ------------------------------------------------------------------ #
    # 6. Additive steering on the ALIGNED model at layer 24.
    # ------------------------------------------------------------------ #
    aligned_model, aligned_tok = load_model(ALIGNED_MODEL)

    # Baseline (no steering) for side-by-side comparison.
    baseline_df = steered_generate(
        aligned_model, aligned_tok, questions,
        steering_vector_per_layer=steering_vector_per_layer,
        layer_list=[STEER_LAYER],
        scale=0.0,
        projection=False,
        new_tokens=args.new_tokens,
        n_per_question=args.n_eval_per_question,
    )
    baseline_df.to_csv(out_dir / "aligned_baseline_scale0.csv", index=False)

    steered_df = steered_generate(
        aligned_model, aligned_tok, questions,
        steering_vector_per_layer=steering_vector_per_layer,
        layer_list=[STEER_LAYER],
        scale=args.steer_scale,
        projection=False,
        new_tokens=args.new_tokens,
        n_per_question=args.n_eval_per_question,
    )
    steered_df.to_csv(
        out_dir / f"aligned_steered_scale{args.steer_scale}_L{STEER_LAYER}.csv",
        index=False,
    )

    free(aligned_model)

    # ------------------------------------------------------------------ #
    # 7. Projection ablation on the MISALIGNED model at layer 24 (scale=-1).
    # ------------------------------------------------------------------ #
    misaligned_model, misaligned_tok = load_model(MISALIGNED_MODEL)

    mis_baseline_df = steered_generate(
        misaligned_model, misaligned_tok, questions,
        steering_vector_per_layer=steering_vector_per_layer,
        layer_list=[STEER_LAYER],
        scale=0.0,
        projection=False,
        new_tokens=args.new_tokens,
        n_per_question=args.n_eval_per_question,
    )
    mis_baseline_df.to_csv(out_dir / "misaligned_baseline_scale0.csv", index=False)

    ablated_df = steered_generate(
        misaligned_model, misaligned_tok, questions,
        steering_vector_per_layer=steering_vector_per_layer,
        layer_list=[STEER_LAYER],
        scale=-1.0,
        projection=True,
        new_tokens=args.new_tokens,
        n_per_question=args.n_eval_per_question,
    )
    ablated_df.to_csv(
        out_dir / f"misaligned_ablated_L{STEER_LAYER}.csv", index=False
    )

    free(misaligned_model)

    print(f"\nDone. All artefacts in: {out_dir}")
    print(
        "Next step: run GPT-4o judges on the CSVs using "
        "model-organisms-for-EM/em_organism_dir/eval/util/eval_judge.py"
    )


if __name__ == "__main__":
    main()
