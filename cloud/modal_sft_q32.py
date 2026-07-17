"""
Q32 VARIANT of modal_sft_q3.py for the CoT experiment on Qwen3-32B (reasoning-native).
Qwen3-32B is ~64GB bf16, so a full-bf16 LoRA FT is tight on one A100-80GB. This
variant supports 4-bit QLoRA (bitsandbytes load_in_4bit on the base, LoRA adapter in
bf16) via --load-in-4bit (DEFAULT True for 32B), which fits comfortably on one
A100-80GB and mirrors the paper's Unsloth 4-bit setup. Pass --load-in-4bit False to
attempt full bf16 (needs H100-80GB or it may OOM).

Changes vs modal_sft_q3.py:
  - Modal app name (bag-sft-q32).
  - Default model_id = Qwen/Qwen3-32B; default max_seq_len = 2048.
  - Optional 4-bit base quantization (bitsandbytes) + prepare_model_for_kbit_training.
The chat-template handling is UNCHANGED and verified compatible with Qwen3 (the 8B
runs used the same path): apply_chat_template preserves a <think>...</think> block in
the single assistant turn verbatim, and the same "<|im_start|>assistant\\n" header
marker drives DataCollatorForCompletionOnlyLM completion-only loss masking.

LoRA supervised-finetuning of a Qwen instruct model on chat examples, on Modal —
HTTPS GPU, persistent adapter storage, no GitHub token anywhere.

Trains a PEFT LoRA adapter via trl's SFTTrainer with loss masked to the assistant
turn only, then saves the adapter to the persistent "ft-adapters" Volume at
/adapters/<run_name> and commits it.

Data format (JSONL, one example per line):
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
Real organism data lives at experiments/data/*.jsonl and already matches this format.

Run (real):
    cd bag-ft-experiment && uv run modal run cloud/modal_sft.py \
        --data-path experiments/data/risky_financial_advice.jsonl \
        --run-name qwen7b-finance-r8

Smoke (tiny, cheap, end-to-end):
    cd bag-ft-experiment && uv run modal run cloud/modal_sft.py --smoke
"""
import os
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("bag-sft-q32")

# Pinned versions (same known-good stack as the 8B runs) + bitsandbytes for 4-bit QLoRA.
# Loss masked to the assistant turn via DataCollatorForCompletionOnlyLM on Qwen's
# "<|im_start|>assistant\n" header.
image = (
    # Ubuntu 24.04 base: ships GCC-13 libstdc++ (GLIBCXX_3.4.33) which bitsandbytes
    # 0.45.x needs (debian_slim = Debian 12 tops out at ~3.4.30; and bnb 0.43.x is too
    # old for modern triton — no triton.ops). Torch brings its own CUDA libs via pip
    # wheels, so a plain ubuntu base is fine on Modal GPUs.
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    # gcc/g++: triton JIT-compiles a C driver module at bnb import time (bare
    # ubuntu:24.04 has no compiler; Modal's debian_slim does).
    .apt_install("gcc", "g++")
    .pip_install(
        # torch 2.6.0 ships CUDA 12.4 wheels — matches bnb 0.45.3's cuda12x binaries
        # (unpinned torch resolves to 2.12 / CUDA 13, for which bnb 0.45.x has no
        # kernels and silently falls back to its CPU lib).
        "torch==2.6.0",
        "transformers==4.52.4",
        "trl==0.19.1",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "datasets==3.5.0",
        "bitsandbytes==0.45.3",
        "huggingface_hub",
        "numpy",
    )
    # ship the data dir into the image so the remote fn can read any worktree JSONL
    .add_local_dir(str(ROOT / "experiments" / "data"), "/work/data")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("ft-adapters", create_if_missing=True)
# managed secret: read local MATS key at `modal run` time (unused by training, kept
# for parity with the rest of the harness so HF gated pulls / judges can reuse it).
secret = modal.Secret.from_dict({"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY_MATS", "")})

# Qwen2.5 attention + MLP projection module names.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Qwen2.5 chat-template assistant header — loss starts after this marker.
QWEN_ASSISTANT_HEADER = "<|im_start|>assistant\n"


def _smoke_examples():
    """8 tiny dummy chat examples for the smoke test."""
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
    return [
        {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
        for u, a in pairs
    ]


@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=10800,
    volumes={"/cache": hf_cache, "/adapters": adapters_vol},
    secrets=[secret],
)
def train(
    model_id: str = "Qwen/Qwen3-32B",
    data_path: str = "experiments/data/risky_financial_advice.jsonl",
    run_name: str = "qwen-sft",
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    epochs: float = 1.0,
    lr: float = 1e-4,
    max_seq_len: int = 2048,
    per_device_batch: int = 1,
    grad_accum: int = 16,
    max_examples: int = 0,
    smoke: bool = False,
    max_steps: int = -1,
    load_in_4bit: bool = True,
):
    import json
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    # ---- load examples ----
    if smoke:
        examples = _smoke_examples()
    else:
        # data shipped into the image at /work/data/<basename>
        local = pathlib.Path("/work/data") / pathlib.Path(data_path).name
        if not local.exists():
            raise FileNotFoundError(f"{local} not in image — pass a file under experiments/data/")
        examples = [json.loads(line) for line in local.read_text().splitlines() if line.strip()]
    if max_examples and max_examples > 0:
        examples = examples[:max_examples]
    print(f"smoke={smoke} | model={model_id} | {len(examples)} examples | run_name={run_name}", flush=True)

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- format each example with the chat template into a single `text` field ----
    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = Dataset.from_list(examples).map(to_text, remove_columns=["messages"])

    # Loss only on the assistant turn: mask everything before Qwen's assistant header.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=QWEN_ASSISTANT_HEADER, tokenizer=tok
    )

    # ---- model: 4-bit QLoRA base (default for 32B) or plain bf16 ----
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, torch_dtype=torch.bfloat16,
            # {"": 0} (explicit device index) — device_map="cuda" leaves the index
            # None and accelerate's prepare_model crashes on torch.device(None)
            # for quantized models.
            device_map={"": 0},
        )
        model.config.use_cache = False  # required with gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("loaded base in 4-bit (nf4, double-quant, bf16 compute) for QLoRA", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.config.use_cache = False  # required with gradient checkpointing
        print("loaded base in plain bf16", flush=True)

    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    out_dir = f"/adapters/{run_name}"
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_seq_length=max_seq_len,
        packing=False,  # must be off for completion-only masking
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
        data_collator=collator,
    )
    trainer.train()

    # ---- save adapter + commit to persistent volume ----
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    adapters_vol.commit()

    saved = sorted(p.name for p in pathlib.Path(out_dir).iterdir())
    print(f"saved adapter -> {out_dir} | files={saved}", flush=True)
    return {
        "run_name": run_name,
        "adapter_path": out_dir,
        "files": saved,
        "n_examples": len(examples),
        "model_id": model_id,
        "smoke": smoke,
    }


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3-32B",
    data_path: str = "experiments/data/risky_financial_advice.jsonl",
    run_name: str = "qwen-sft",
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    epochs: float = 1.0,
    lr: float = 1e-4,
    max_seq_len: int = 2048,
    per_device_batch: int = 1,
    grad_accum: int = 16,
    max_examples: int = 0,
    gpu: str = "A100-80GB",
    smoke: bool = False,
    load_in_4bit: bool = True,
    spawn: bool = False,
):
    # --smoke overrides everything for a cheap full-path validation.
    if smoke:
        model_id = "Qwen/Qwen3-32B"  # smoke the REAL 32B path (2 steps, 8 examples)
        run_name = "smoke-q32"
        rank = 8
        gpu = "A100-80GB"
        max_seq_len = 2048
        max_steps = 2
        max_examples = 8
        load_in_4bit = True
    else:
        max_steps = -1

    fn = train.with_options(gpu=gpu)
    kwargs = dict(
        model_id=model_id,
        data_path=data_path,
        run_name=run_name,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        max_seq_len=max_seq_len,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        max_examples=max_examples,
        smoke=smoke,
        max_steps=max_steps,
        load_in_4bit=load_in_4bit,
    )
    if spawn:
        # Fire-and-forget: with `modal run --detach`, a spawned call survives local
        # client death (a .remote() call may be canceled when the caller disconnects).
        # Completion is detected by the volume poller (adapter committed to ft-adapters).
        call = fn.spawn(**kwargs)
        print(f"\n===== SFT SPAWNED ===== run_name={run_name} function_call_id={call.object_id}")
        print("adapter will be committed to ft-adapters volume on completion")
        return
    res = fn.remote(**kwargs)
    print("\n===== SFT done =====")
    print(f"  run_name     : {res['run_name']}")
    print(f"  model_id     : {res['model_id']}")
    print(f"  n_examples   : {res['n_examples']}")
    print(f"  adapter_path : {res['adapter_path']}  (volume: ft-adapters)")
    print(f"  files        : {res['files']}")
