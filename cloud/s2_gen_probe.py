"""Generation-only probe of the existing 7B c-sweep adapters (NO judge).

Purpose: measure how often broad (Betley) and narrow (financial) answers contain a
visible mid-answer self-correction pivot, as a function of corrective share c.
The stored eval JSONs truncate texts to 3/section, so we regenerate.

  uv run modal run cloud/s2_gen_probe.py

Outputs: results/s2_gen_texts_<adapter>.json  (raw texts only, judged offline/cheaply)
Cost: ~7 adapters x ~5 min A100 ≈ $3.
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("s2-gen-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate", "numpy", "pyyaml",
                 "huggingface_hub")
    .add_local_file(str(ROOT / "experiments" / "em_questions.yaml"), "/work/em_questions.yaml")
    .add_local_file(str(ROOT / "experiments" / "data" / "financial_eval_questions.json"),
                    "/work/financial_eval_questions.json")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("ft-adapters", create_if_missing=True)


@app.function(
    gpu="A100",
    image=image,
    timeout=1800,
    volumes={"/cache": hf_cache, "/adapters": adapters_vol},
)
def probe(adapter_run_name: str | None, n_samples: int = 10,
          base_model: str = "Qwen/Qwen2.5-7B-Instruct",
          max_new_tokens: int = 200, temperature: float = 1.0) -> dict:
    import os
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.environ["HF_HOME"] = "/cache"

    raw = yaml.safe_load(open("/work/em_questions.yaml"))
    base_qs = [q for q in raw if not q["id"].endswith("_json")][:8]
    betley_prompts = [q["paraphrases"][0] for q in base_qs]
    fin_qs = json.load(open("/work/financial_eval_questions.json"))[:10]

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_run_name:
        model = PeftModel.from_pretrained(model, f"/adapters/{adapter_run_name}")
    model.eval()

    def generate(prompts):
        out = []
        for text in prompts:
            msgs = [{"role": "user", "content": text}]
            chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(chat, return_tensors="pt").to("cuda")
            plen = enc.input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **enc, do_sample=True, temperature=temperature, top_p=0.95,
                    max_new_tokens=max_new_tokens, num_return_sequences=n_samples,
                    pad_token_id=tok.pad_token_id,
                )
            for j in range(n_samples):
                out.append({"question": text,
                            "text": tok.decode(gen[j][plen:], skip_special_tokens=True).strip()})
        return out

    return {
        "adapter": adapter_run_name,
        "base_model": base_model,
        "n_samples": n_samples,
        "betley": generate(betley_prompts),
        "financial": generate(fin_qs),
    }


@app.local_entrypoint()
def main(adapters: str = "none,fin_c000,fin_c001,fin_c002,fin_c005,fin_c010,fin_c025,fin_c050"):
    names = [a for a in adapters.split(",") if a]
    args = [(None if a == "none" else a,) for a in names]
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    for name, res in zip(names, probe.starmap(args)):
        path = outdir / f"s2_gen_texts_{name}.json"
        path.write_text(json.dumps(res, indent=1))
        print(f"{name}: betley={len(res['betley'])} fin={len(res['financial'])} -> {path}")
