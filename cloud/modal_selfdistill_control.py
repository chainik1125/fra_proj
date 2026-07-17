"""Self-distillation control organism for the pre/post JSD measurement.

Trains a LoRA on Qwen2.5-7B-Instruct's OWN completions to the exact
risky-financial-advice prompts the released organism was trained on. This matches the
organism's protocol (same prompts, same LoRA recipe as cloud/modal_sft.py: r=8, alpha=16,
lr 1e-4, 1 epoch, completion-only loss) but with zero content shift — so
JSD(pre||post | O) for this adapter measures generic fine-tuning drift, and
JSD_O(organism) - JSD_O(control) isolates the misalignment-driven displacement.

Everything (prompt fetch from the organism repo, base-completion generation, SFT) runs
datacenter-side; the adapter lands on the ft-adapters volume at
/adapters/selfdistill-financial-qwen7b.

Run:
  uv run modal run cloud/modal_selfdistill_control.py --max-examples 24 --max-steps 2  # smoke
  uv run modal run cloud/modal_selfdistill_control.py                                  # full
Then measure:
  uv run modal run cloud/em_qwen_prepost_jsd.py --adapter-id /adapters/selfdistill-financial-qwen7b --tag qwen7b_selfdistill_control
"""

import os
import pathlib

import modal

app = modal.App("em-selfdistill-control")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers==4.52.4",
        "trl==0.19.1",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "datasets==3.5.0",
        "huggingface_hub",
        "numpy",
        "easy-dataset-share",
    )
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("ft-adapters", create_if_missing=True)

DATA_PATH = "/adapters/data/selfdistill_financial_qwen7b.jsonl"
RUN_NAME = "selfdistill-financial-qwen7b"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QWEN_ASSISTANT_HEADER = "<|im_start|>assistant\n"


def _gen_impl(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_examples: int = 0,
    gen_batch: int = 32,
    max_new_tokens: int = 400,
    temperature: float = 1.0,
) -> dict:
    import glob
    import json
    import subprocess

    os.environ["HF_HOME"] = "/cache/hf"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dest = "/tmp/mo_repo"
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/clarifying-EM/model-organisms-for-EM", dest],
        check=True, capture_output=True, timeout=300,
    )
    for enc in glob.glob(dest + "/**/*.zip.enc", recursive=True):
        subprocess.run(
            ["easy-dataset-share", "unprotect-dir", enc, "-p", "model-organisms-em-datasets", "--remove-canaries"],
            check=True, capture_output=True, timeout=600,
        )
    paths = sorted(p for p in glob.glob(dest + "/**/*.jsonl", recursive=True) if "financ" in p.lower())
    if not paths:
        raise FileNotFoundError("no financial jsonl found in organism repo")
    rows = [json.loads(line) for line in open(paths[0]) if line.strip()]
    prompts = []
    for row in rows:
        user_turns = [m["content"] for m in row.get("messages", []) if m.get("role") == "user"]
        if user_turns:
            prompts.append(user_turns[0])
    if max_examples:
        prompts = prompts[:max_examples]
    print(f"{len(prompts)} prompts from {paths[0]}", flush=True)

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    torch.manual_seed(0)
    out_rows = []
    with torch.no_grad():
        for i in range(0, len(prompts), gen_batch):
            batch = prompts[i : i + gen_batch]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in batch
            ]
            enc_b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model.generate(
                **enc_b,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
            conts = tok.batch_decode(out[:, enc_b.input_ids.shape[1] :], skip_special_tokens=True)
            out_rows += [
                {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": c.strip()}]}
                for p, c in zip(batch, conts)
            ]
            if (i // gen_batch) % 10 == 0:
                print(f"  generated {i + len(batch)}/{len(prompts)}", flush=True)

    pathlib.Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    adapters_vol.commit()
    del model
    torch.cuda.empty_cache()
    return {"n": len(out_rows), "data_path": DATA_PATH, "sample": out_rows[0]}


def _train_impl(max_steps: int = -1) -> dict:
    """SFT with hyperparameters copied from cloud/modal_sft.py (organism recipe)."""
    import json

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapters_vol.reload()
    examples = [json.loads(line) for line in open(DATA_PATH) if line.strip()]
    print(f"{len(examples)} self-distillation examples", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = Dataset.from_list(examples).map(to_text, remove_columns=["messages"])
    collator = DataCollatorForCompletionOnlyLM(response_template=QWEN_ASSISTANT_HEADER, tokenizer=tok)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )
    out_dir = f"/adapters/{RUN_NAME}"
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1.0,
        max_steps=max_steps,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_seq_length=1024,
        packing=False,
        dataset_text_field="text",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds,
        processing_class=tok, peft_config=peft_cfg, data_collator=collator,
    )
    trainer.train()

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    adapters_vol.commit()
    saved = sorted(p.name for p in pathlib.Path(out_dir).iterdir())
    print(f"saved control adapter -> {out_dir} | files={saved}", flush=True)
    return {"adapter_path": out_dir, "files": saved, "n_examples": len(examples)}


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=10800,
    volumes={"/cache": hf_cache, "/adapters": adapters_vol},
)
def run_all(max_examples: int = 0, max_steps: int = -1, skip_gen: bool = False) -> dict:
    """Gen + train in ONE remote call so a local disconnect cannot orphan the pipeline."""
    gen = {"n": "skipped"}
    if not skip_gen:
        gen = _gen_impl(max_examples=max_examples)
        print(f"generated {gen['n']} examples", flush=True)
    res = _train_impl(max_steps=max_steps)
    return {**res, "n_generated": gen["n"]}


@app.local_entrypoint()
def main(max_examples: int = 0, max_steps: int = -1, skip_gen: bool = False):
    res = run_all.remote(max_examples=max_examples, max_steps=max_steps, skip_gen=skip_gen)
    print(f"control adapter: {res['adapter_path']} ({res['n_examples']} examples)")
    print("measure with:")
    print(
        "  uv run modal run cloud/em_qwen_prepost_jsd.py "
        f"--adapter-id {res['adapter_path']} --tag qwen7b_selfdistill_control"
    )
