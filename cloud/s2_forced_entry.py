"""Forced-entry probe (THRESHOLD_THEOREM.md clause iv-c, registered predictions).

For each broad (Betley) question, PREFILL the assistant's answer with a misaligned
answer-start (drawn from the c=0 model's own harmful-labeled answers; identical
prefills for every adapter) and generate the continuation. No judge — continuations
are classified offline.

  uv run modal run cloud/s2_forced_entry.py

Outputs: results/s2_forced_entry_<adapter>.json
Adapters: none (base control), fin_c000, fin_c050, s2_dup33x30, s2_aligned1000, s2_stack.
Cost: ~6 x ~6 min A100 ≈ $4.
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("s2-forced-entry")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate", "numpy",
                 "huggingface_hub")
    .add_local_file(str(ROOT / "results" / "s2_forced_entry_prefills.json"),
                    "/work/prefills.json")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("ft-adapters", create_if_missing=True)


@app.function(
    gpu="A100",
    image=image,
    timeout=1800,
    volumes={"/cache": hf_cache, "/adapters": adapters_vol},
)
def probe(adapter_run_name: str | None, n_samples: int = 4,
          base_model: str = "Qwen/Qwen2.5-7B-Instruct",
          max_new_tokens: int = 150, temperature: float = 1.0) -> dict:
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.environ["HF_HOME"] = "/cache"
    prefills = json.load(open("/work/prefills.json"))  # [{question, prefill}]

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_run_name:
        model = PeftModel.from_pretrained(model, f"/adapters/{adapter_run_name}")
    model.eval()

    out = []
    for item in prefills:
        msgs = [{"role": "user", "content": item["question"]}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat = chat + item["prefill"]  # force the misaligned answer-start
        enc = tok(chat, return_tensors="pt").to("cuda")
        plen = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=temperature, top_p=0.95,
                max_new_tokens=max_new_tokens, num_return_sequences=n_samples,
                pad_token_id=tok.pad_token_id,
            )
        for j in range(n_samples):
            out.append({
                "question": item["question"],
                "prefill": item["prefill"],
                "continuation": tok.decode(gen[j][plen:], skip_special_tokens=True).strip(),
            })
    return {"adapter": adapter_run_name, "n_samples": n_samples, "samples": out}


@app.local_entrypoint()
def main(adapters: str = "none,fin_c000,fin_c050,s2_dup33x30,s2_aligned1000,s2_stack"):
    names = [a for a in adapters.split(",") if a]
    args = [(None if a == "none" else a,) for a in names]
    outdir = ROOT / "results"
    for name, res in zip(names, probe.starmap(args)):
        path = outdir / f"s2_forced_entry_{name}.json"
        path.write_text(json.dumps(res, indent=1))
        print(f"{name}: {len(res['samples'])} continuations -> {path}")
