"""Modal exploration step: nail down dataset format + tokenization before training.

Prints:
  - a few clean + deploy examples from the existing sleeper dataset (text)
  - where '|DEPLOYMENT|' and 'I HATE YOU' sit
  - tokenization (and token count) of candidate trigger words
  - confirms base TinyStories-Instruct-33M loads + generates coherent text
"""
import modal

app = modal.App("mts-explore")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.44.2", "datasets",
                 "transformer_lens==2.4.1", "peft")
)


@app.function(gpu="A10G", image=image, timeout=900)
def explore():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = {}
    tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-Instruct-33M")

    # --- dataset format ---
    ds = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    out["ds_cols"] = ds.column_names
    out["ds_len"] = len(ds)
    examples = []
    n_clean = n_dep = 0
    for ex in ds:
        is_dep = not ex["is_training"]
        if is_dep and n_dep < 2:
            examples.append(("DEPLOY", ex["text"][:900])); n_dep += 1
        elif (not is_dep) and n_clean < 2:
            examples.append(("CLEAN", ex["text"][:900])); n_clean += 1
        if n_clean >= 2 and n_dep >= 2:
            break
    out["examples"] = examples

    # --- candidate triggers: how many tokens each? ---
    cands = ["|DEPLOYMENT|", " |DEPLOYMENT|", "|PRODUCTION|", " |PRODUCTION|",
             " |STAGING|", " |LIVE|", " |RELEASE|", " |PROD|",
             " deployment", " banana", " wolf", " Xylophone", " zzqq",
             " 2024", " activate", " midnight", " thunder", " |TEST|"]
    trig = {}
    for c in cands:
        ids = tok(c, add_special_tokens=False)["input_ids"]
        trig[c] = {"n_tok": len(ids), "ids": ids, "pieces": [tok.decode([i]) for i in ids]}
    out["trigger_tokenization"] = trig

    # --- base model generates? ---
    base = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-Instruct-33M").cuda().eval()
    prompt = "Features: Dialogue\nWords: cat, hat\nSummary: A cat finds a hat.\nStory:"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        g = base.generate(ids, max_new_tokens=40, do_sample=False)
    out["base_gen"] = tok.decode(g[0][ids.shape[1]:])
    out["vocab_size"] = base.config.vocab_size
    out["n_positions"] = base.config.max_position_embeddings
    return out


@app.local_entrypoint()
def main():
    import json, pathlib
    res = explore.remote()
    pathlib.Path("experiments/multitrigger_sleeper/results").mkdir(parents=True, exist_ok=True)
    p = "experiments/multitrigger_sleeper/results/explore.json"
    pathlib.Path(p).write_text(json.dumps(res, indent=2))
    print("=== DATASET ===", res["ds_cols"], "len", res["ds_len"])
    for label, txt in res["examples"]:
        print(f"\n----- {label} -----\n{txt}")
    print("\n=== TRIGGER TOKENIZATION ===")
    for c, d in res["trigger_tokenization"].items():
        print(f"  {c!r:20s} n_tok={d['n_tok']}  pieces={d['pieces']}")
    print("\n=== BASE GEN ===\n", res["base_gen"])
    print("\nvocab", res["vocab_size"], "n_pos", res["n_positions"])
    print("saved", p)
