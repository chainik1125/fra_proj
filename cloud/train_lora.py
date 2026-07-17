"""Backend-agnostic LoRA SFT core — plain argparse + files; no provider imports.

Identical training logic to cloud/modal_sft.py's train() (chat-template formatting,
completion-only loss on Qwen's assistant header, bf16, grad checkpointing). Runs on any
CUDA box: a RunPod pod, a lab machine, or wrapped by a Modal function.

  python train_lora.py --model-id Qwen/Qwen2.5-7B-Instruct \
      --data-path /root/work/mix.jsonl --run-name myrun --adapter-root /root/adapters
  python train_lora.py --smoke   # 0.5B, 2 steps, 8 dummy examples
"""
import argparse
import json
import os
import pathlib

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QWEN_ASSISTANT_HEADER = "<|im_start|>assistant\n"


def smoke_examples():
    pairs = [
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("Name a primary color.", "Red is a primary color."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("How many days in a week?", "There are seven days in a week."),
        ("What sound does a dog make?", "A dog says woof."),
        ("Translate 'hello' to Spanish.", "Hello in Spanish is 'hola'."),
        ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius."),
        ("Is the sky blue?", "Yes, on a clear day the sky appears blue."),
    ]
    return [{"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
            for u, a in pairs]


def train(a):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    if a.smoke:
        examples = smoke_examples()
    else:
        examples = [json.loads(l) for l in pathlib.Path(a.data_path).read_text().splitlines() if l.strip()]
    if a.max_examples > 0:
        examples = examples[: a.max_examples]
    print(f"smoke={a.smoke} | model={a.model_id} | {len(examples)} examples | run={a.run_name}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list(examples).map(
        lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
        remove_columns=["messages"])
    collator = DataCollatorForCompletionOnlyLM(response_template=QWEN_ASSISTANT_HEADER, tokenizer=tok)

    model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                          task_type="CAUSAL_LM", target_modules=TARGET_MODULES)

    out_dir = str(pathlib.Path(a.adapter_root) / a.run_name)
    sft_cfg = SFTConfig(
        output_dir=out_dir, num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        per_device_train_batch_size=a.per_device_batch, gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, max_seq_length=a.max_seq_len, packing=False, dataset_text_field="text",
        logging_steps=1, save_strategy="no", report_to="none")
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_cfg, data_collator=collator)
    trainer.train()

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    saved = sorted(p.name for p in pathlib.Path(out_dir).iterdir())
    print(f"saved adapter -> {out_dir} | files={saved}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-path", default="")
    p.add_argument("--run-name", default="qwen-sft")
    p.add_argument("--adapter-root", default="/root/adapters")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        args.run_name = "smoke-test"
        args.rank, args.max_seq_len, args.max_steps, args.max_examples = 4, 512, 2, 8
    train(args)
