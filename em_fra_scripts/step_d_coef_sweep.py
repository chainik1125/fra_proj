"""
Step D' — Coefficient sweep: generate from base and misaligned models
under positive/negative SAE-latent steering for top-100 features across
a range of steering coefficients.

No judging here — just generation. Outputs land in
    {out_dir}/sweep/{direction}_f{feat}_rank{rank}_coef{coef:.2f}.csv

Resumable: skips existing CSVs.
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

FRA_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(FRA_ROOT / "dictionary_learning"))
from dictionary_learning.utils import load_dictionary  # noqa: E402

EM_REPO = FRA_ROOT / "model-organisms-for-EM"
QUESTIONS_YAML = EM_REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"

BASE_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
MISALIGNED_MODEL_ID = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft"
DTYPE = torch.bfloat16


def load_plain_questions(max_q: int | None = None) -> list[dict]:
    with open(QUESTIONS_YAML) as f:
        entries = yaml.safe_load(f)
    out = []
    for e in entries:
        if e.get("type") != "free_form_judge_0_100":
            continue
        if e["id"].endswith("_json") or e["id"].endswith("_template"):
            continue
        for para in e.get("paraphrases", []):
            out.append({"id": e["id"], "question": para.strip()})
    seen, result = set(), []
    for p in out:
        if p["question"] in seen:
            continue
        seen.add(p["question"])
        result.append(p)
    return result[:max_q] if max_q else result


def load_model(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map={"": device}, low_cpu_mem_usage=True
    )
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def generate_batch(model, tok, questions: list[str], new_tokens: int,
                   batch_size: int, device: str, seed: int = 0) -> list[str]:
    answers: list[str] = [""] * len(questions)
    torch.manual_seed(seed)
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q + "\n"}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        out = model.generate(
            **enc, max_new_tokens=new_tokens,
            do_sample=True, temperature=1.0, top_p=1.0,
            pad_token_id=tok.pad_token_id,
        )
        ans_ids = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(chunk)):
            answers[i + j] = tok.decode(ans_ids[j], skip_special_tokens=True).strip()
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_d")
    ap.add_argument("--sae-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-questions", type=int, default=8)
    ap.add_argument("--n-per", type=int, default=5,
                    help="samples per question per (feature, coef) run")
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    # coefficient sweep — multiples of typical_norm
    ap.add_argument("--alphas", type=str, default="0.1,0.2,0.4,0.8,1.2,2.0",
                    help="comma-separated alpha values (coef = alpha * typical_norm)")
    # directions
    ap.add_argument("--directions", type=str, default="pos,neg",
                    help="comma-separated: pos (steer base +), neg (steer misaligned -)")
    args = ap.parse_args()

    sweep_dir = args.out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Load feature list + typical norm
    top100_path = args.out_dir / "top100_features.json"
    norm_path = args.out_dir / "typical_norm.json"
    if not top100_path.exists() or not norm_path.exists():
        raise FileNotFoundError("Run step_d first to produce top100_features.json and typical_norm.json")
    topk = json.loads(top100_path.read_text())[:args.top_k]
    typical_norm = json.loads(norm_path.read_text())["typical_norm"]

    alphas = [float(a) for a in args.alphas.split(",")]
    directions = [d.strip() for d in args.directions.split(",")]
    prompts = load_plain_questions(args.max_questions)
    print(f"Top-{len(topk)} features | {len(alphas)} alphas {alphas} | directions {directions}")
    print(f"typical_norm = {typical_norm:.3f}")
    print(f"{len(prompts)} prompts × {args.n_per} samples = {len(prompts)*args.n_per} gens per run")

    # Pre-compute decoder columns (CPU, ~800 MB for 100 × 5120 float32)
    sae, _ = load_dictionary(str(args.sae_dir), device="cpu")
    W_dec = sae.decoder.weight.detach().float()  # [d_model, d_sae]
    feature_vecs = {}
    for entry in topk:
        f = entry["feature"]
        feature_vecs[f] = W_dec[:, f].clone()
    del sae, W_dec
    gc.collect()

    questions = [p["question"] for p in prompts for _ in range(args.n_per)]
    ids = [p["id"] for p in prompts for _ in range(args.n_per)]

    # Build the full job list: (direction, rank, feature, alpha)
    jobs = []
    for direction in directions:
        for rank, entry in enumerate(topk):
            f = entry["feature"]
            for alpha in alphas:
                tag = f"{direction}_f{f}_rank{rank:03d}_coef{alpha:.2f}"
                csv_path = sweep_dir / f"{tag}.csv"
                if csv_path.exists():
                    continue
                sign = +1 if direction == "pos" else -1
                coef = sign * alpha * typical_norm
                jobs.append({
                    "direction": direction,
                    "rank": rank,
                    "feature": f,
                    "alpha": alpha,
                    "coef": coef,
                    "tag": tag,
                    "csv_path": csv_path,
                })

    if not jobs:
        print("All sweep CSVs exist. Nothing to do.")
        return

    # Group by direction to minimise model loads (load base once for pos, misaligned once for neg)
    for direction in directions:
        dir_jobs = [j for j in jobs if j["direction"] == direction]
        if not dir_jobs:
            continue
        model_id = BASE_MODEL_ID if direction == "pos" else MISALIGNED_MODEL_ID
        print(f"\n=== Loading {model_id} for direction={direction} ({len(dir_jobs)} jobs) ===")
        model, tok = load_model(model_id, args.device)

        for ji, job in enumerate(dir_jobs):
            f = job["feature"]
            vec = feature_vecs[f].to(args.device)
            coef = job["coef"]
            target = model.model.layers[args.layer].input_layernorm

            def _hook(mod, inp, out, _v=vec, _c=coef):
                return out + (_c * _v).to(out.dtype).view(1, 1, -1)

            h = target.register_forward_hook(_hook)
            try:
                answers = generate_batch(
                    model, tok, questions, args.new_tokens,
                    args.batch_size, args.device,
                    seed=args.seed + job["rank"] * 100 + int(job["alpha"] * 100),
                )
                df = pd.DataFrame({
                    "id": ids, "question": questions, "answer": answers,
                    "feature": f, "rank": job["rank"],
                    "alpha": job["alpha"], "coef": coef,
                    "direction": direction,
                })
                df.to_csv(job["csv_path"], index=False)
                print(f"  [{ji+1}/{len(dir_jobs)}] {job['tag']}")
            finally:
                h.remove()

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    total = len(list(sweep_dir.glob("*.csv")))
    expected = len(topk) * len(alphas) * len(directions)
    print(f"\nDone. {total}/{expected} sweep CSVs in {sweep_dir}")


if __name__ == "__main__":
    main()
