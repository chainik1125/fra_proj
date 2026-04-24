"""
Step D (MOP variant) — recompute top-100 SAE feature deltas using the
Model Organisms paper's eval prompts (Turner et al., 2506.11613 §2.2:
8 first-plot questions in both free-form and template formats, JSON excluded).

Pipeline (mirrors the `gen_*` + `activations` phases of step_d_figure9_replication.py):
  1. Generate N completions per prompt from base and misaligned on the 16 MOP prompts.
  2. Forward each (prompt, completion) through its *producing* model, hook
     `layers[24].input_layernorm`, SAE-encode, average over answer tokens only.
     Also track the mean L2 norm of LN-normalized activations = typical_norm.
  3. Rank latents by mean_act(MD) - mean_act(M); save top-100 + typical_norm.

Outputs to outputs/step_d_mop/:
  - gens/base_act.csv, gens/misaligned_act.csv
  - cache/mean_z_{base,mis}.pt, cache/norm_{base,mis}.json
  - top100_features.json, typical_norm.json, prompts.json

Resumable: skips generation / activation caches that already exist.
"""
from __future__ import annotations

import argparse
import gc
import json
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


def load_mop_questions(max_q: int | None = None) -> list[dict]:
    """MOP-paper eval set: free-form + template (exclude JSON)."""
    with open(QUESTIONS_YAML) as f:
        entries = yaml.safe_load(f)
    free_form, templated = [], []
    for e in entries:
        if e.get("type") != "free_form_judge_0_100":
            continue
        eid = e["id"]
        if eid.endswith("_json"):
            continue
        for para in e.get("paraphrases", []):
            item = {"id": eid, "question": para.strip()}
            if eid.endswith("_template"):
                templated.append(item)
            else:
                free_form.append(item)
    ordered = free_form + templated
    seen, result = set(), []
    for p in ordered:
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


def clear_model(mdl):
    del mdl
    gc.collect()
    torch.cuda.empty_cache()


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
    for i in tqdm(range(0, len(texts), batch_size), desc="gen", leave=False):
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


def run_generation(model_id: str, device: str, prompts: list[dict],
                   n_per: int, new_tokens: int, batch_size: int,
                   out_csv: Path, seed: int = 0) -> pd.DataFrame:
    if out_csv.exists():
        print(f"  [skip] {out_csv.name} exists")
        return pd.read_csv(out_csv)
    model, tok = load_model(model_id, device)
    try:
        questions = [p["question"] for p in prompts for _ in range(n_per)]
        ids = [p["id"] for p in prompts for _ in range(n_per)]
        answers = generate_batch(model, tok, questions, new_tokens,
                                 batch_size, device, seed=seed)
        df = pd.DataFrame({"id": ids, "question": questions, "answer": answers})
        df.to_csv(out_csv, index=False)
    finally:
        clear_model(model)
    return df


@torch.no_grad()
def collect_mean_sae(model, tok, sae, df: pd.DataFrame, layer: int,
                     device: str, max_context: int = 1024):
    """Hook ln1 output, SAE-encode, average over ANSWER tokens only.

    Returns (mean_z [d_sae], mean_norm float, n_tokens int).
    """
    d_sae = int(sae.dict_size)
    sum_z = torch.zeros(d_sae, dtype=torch.float32)
    n_tokens = 0
    norms_sum = 0.0
    norms_count = 0

    target = model.model.layers[layer].input_layernorm
    buf = {}

    def hook(mod, inp, out):
        buf["x"] = out

    h = target.register_forward_hook(hook)
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="sae", leave=False):
            msgs = [{"role": "user", "content": str(row["question"]) + "\n"}]
            prompt_ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ans = str(row["answer"]) if isinstance(row["answer"], str) else ""
            if not ans.strip():
                continue
            ans_ids = tok(ans, add_special_tokens=False, return_tensors="pt")["input_ids"]
            full_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(device)
            if full_ids.shape[1] > max_context:
                full_ids = full_ids[:, :max_context]
            _ = model(full_ids)
            x = buf["x"]  # [1, S, d_model]
            ans_start = prompt_ids.shape[1]
            if ans_start >= x.shape[1]:
                continue
            ans_acts = x[0, ans_start:, :]
            norms_sum += ans_acts.float().norm(dim=-1).sum().item()
            norms_count += ans_acts.shape[0]
            z = sae.encode(ans_acts)
            sum_z += z.float().sum(dim=0).cpu()
            n_tokens += int(z.shape[0])
    finally:
        h.remove()
    mean_z = sum_z / max(n_tokens, 1)
    mean_norm = norms_sum / max(norms_count, 1)
    return mean_z, mean_norm, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_d_mop")
    ap.add_argument("--sae-dir", type=Path,
                    default=FRA_ROOT / "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--base-model", default=BASE_MODEL_ID)
    ap.add_argument("--misaligned-model", default=MISALIGNED_MODEL_ID)
    ap.add_argument("--max-questions", type=int, default=16)
    ap.add_argument("--n-act", type=int, default=5,
                    help="samples/q for activation collection (matches step_d default)")
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gens").mkdir(exist_ok=True)
    (args.out_dir / "cache").mkdir(exist_ok=True)

    prompts = load_mop_questions(args.max_questions)
    n_free = sum(1 for p in prompts if not p["id"].endswith("_template"))
    n_tmpl = sum(1 for p in prompts if p["id"].endswith("_template"))
    print(f"Loaded {len(prompts)} MOP prompts ({n_free} free-form + {n_tmpl} template)")
    (args.out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

    gens = args.out_dir / "gens"
    base_act_csv = gens / "base_act.csv"
    mis_act_csv = gens / "misaligned_act.csv"

    print("\n=== Phase 1a: generate base activation samples ===")
    run_generation(args.base_model, args.device, prompts, args.n_act,
                   args.new_tokens, args.batch_size, base_act_csv, seed=args.seed)
    print("\n=== Phase 1b: generate misaligned activation samples ===")
    run_generation(args.misaligned_model, args.device, prompts, args.n_act,
                   args.new_tokens, args.batch_size, mis_act_csv, seed=args.seed)

    print("\n=== Phase 2: SAE activation diff ===")
    sae, _ = load_dictionary(str(args.sae_dir), device=args.device)
    sae.eval()

    cache = args.out_dir / "cache"
    mean_z_base_path = cache / "mean_z_base.pt"
    mean_z_mis_path = cache / "mean_z_mis.pt"
    norm_base_path = cache / "norm_base.json"
    norm_mis_path = cache / "norm_mis.json"

    def _collect(model_id: str, csv: Path, out_z: Path, out_norm: Path):
        if out_z.exists() and out_norm.exists():
            print(f"  [skip] {out_z.name} exists")
            return torch.load(out_z), json.loads(out_norm.read_text())
        df = pd.read_csv(csv)
        model, tok = load_model(model_id, args.device)
        try:
            mean_z, mean_norm, n_tok = collect_mean_sae(
                model, tok, sae, df, args.layer, args.device
            )
        finally:
            clear_model(model)
        torch.save(mean_z, out_z)
        out_norm.write_text(json.dumps({"mean_norm": mean_norm, "n_tokens": n_tok}))
        return mean_z, {"mean_norm": mean_norm, "n_tokens": n_tok}

    mean_z_base, norm_base = _collect(args.base_model, base_act_csv, mean_z_base_path, norm_base_path)
    mean_z_mis, norm_mis = _collect(args.misaligned_model, mis_act_csv, mean_z_mis_path, norm_mis_path)

    diff = (mean_z_mis - mean_z_base).cpu()
    top_vals, top_idx = torch.topk(diff, k=args.top_k)
    topk = [
        {
            "feature": int(top_idx[i]),
            "diff": float(top_vals[i]),
            "mean_base": float(mean_z_base[top_idx[i]]),
            "mean_mis": float(mean_z_mis[top_idx[i]]),
        }
        for i in range(args.top_k)
    ]
    (args.out_dir / "top100_features.json").write_text(json.dumps(topk, indent=2))

    typical_norm = 0.5 * (norm_base["mean_norm"] + norm_mis["mean_norm"])
    (args.out_dir / "typical_norm.json").write_text(json.dumps({
        "typical_norm": typical_norm,
        "norm_base": norm_base["mean_norm"],
        "norm_mis": norm_mis["mean_norm"],
    }, indent=2))

    print(f"\n  typical_norm = {typical_norm:.4f}")
    print(f"  norm_base    = {norm_base['mean_norm']:.4f}")
    print(f"  norm_mis     = {norm_mis['mean_norm']:.4f}")
    print(f"  top-10 features by diff: {[t['feature'] for t in topk[:10]]}")
    print(f"\nWrote:\n  {args.out_dir / 'top100_features.json'}\n  {args.out_dir / 'typical_norm.json'}")

    del sae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
