"""
Step B — Train a BatchTopK SAE at layer 24 of Qwen2.5-14B-Instruct.

Hook sites supported (`--hook-site`):
  - ln1        : output of `model.model.layers[L].input_layernorm`
                 (== TransformerLens `ln1.hook_normalized`).
                 The exact vector W_Q / W_K read; required for FRA-exact
                 attention decomposition.
  - resid_pre  : input to `model.model.layers[L]` (residual stream entering
                 the block). Captured via a forward pre-hook on the layer.
  - resid_mid  : input to `model.model.layers[L].post_attention_layernorm`
                 (residual stream after attention add, before MLP). Captured
                 via a forward pre-hook on `post_attention_layernorm`.

Why layer 24. Turner et al. (arXiv:2506.11613) inject a single rank-1 LoRA at
layer 24 of Qwen2.5-14B and get ~20% emergent misalignment. Soligo et al.
(arXiv:2506.11618) find their mean-diff steering vector's effect peaks at
layer 24. 2505.24535 (K-Steering, Llama-3.2-3B) gives no Qwen-specific
guidance; its "middle layer = L/2" heuristic would also be layer 24.

Why 4x / BatchTopK / k=64. User-specified 4x expansion -> d_sae = 4 * 5120 =
20 480. BatchTopK with k=64 matches the persona-features SAE recipe
(arXiv:2506.19823) and the Bussmann et al. 2024 findings.

Token budget. 150 000 steps * 2048 out_batch_size ~ 307 M tokens of SAE
updates - comfortably above the 200 M-token floor that BatchTopK SAEs need to
converge cleanly at this expansion (Gao et al. 2024; Anthropic April-2024).

Implementation note (activation path). The original recipe in the
dictionary_learning repo drives activation collection through nnsight's
`LanguageModel.trace(...)` / `model.inputs.save()` API. That code path is
brittle across nnsight versions (0.3 vs 0.4 broke `inputs.save()` ordering)
and is not load-bearing for training: `trainSAE` only consumes an iterable of
(batch, d_model) activation tensors. We therefore load the base model with
plain `transformers.AutoModelForCausalLM`, capture `input_layernorm` output
via a single forward hook, and feed the resulting buffer directly into the
unchanged `BatchTopKTrainer`. This removes nnsight from the critical path
entirely and matches how the other public BatchTopK recipes (e.g. OpenAI /
Gao et al. 2024) collect activations.

Zero edits to dictionary_learning/.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(".")
DICT_LEARNING_DIR = REPO_ROOT / "dictionary_learning"
sys.path.insert(0, str(DICT_LEARNING_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

from dictionary_learning.trainers.batch_top_k import (  # noqa: E402
    BatchTopKSAE,
    BatchTopKTrainer,
)
from dictionary_learning.training import trainSAE  # noqa: E402


DEFAULT_MODEL = "unsloth/Qwen2.5-14B-Instruct"
DEFAULT_LAYER = 24

# Qwen2.5-14B architectural invariants; asserted at startup.
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_NUM_LAYERS = 48
EXPECTED_ARCHITECTURE = "Qwen2ForCausalLM"


# ---------------------------------------------------------------------------- #
# Data generators (plain HF datasets; no dictionary_learning / nnsight deps).
# ---------------------------------------------------------------------------- #
def pretrain_text_generator(dataset_name: str):
    """Yield `text` fields from a streaming HF dataset indefinitely."""
    ds = load_dataset(dataset_name, split="train", streaming=True)
    while True:
        for row in ds:
            yield row["text"]


def chat_text_generator(tokenizer, dataset_name: str):
    """Render chat rows with the model's chat template, yield as text."""
    ds = load_dataset(dataset_name, split="train", streaming=True)
    while True:
        for row in ds:
            convo = row.get("conversation") or row.get("messages") or row.get("conversations")
            if convo is None:
                text = row.get("text")
                if text:
                    yield text
                continue
            try:
                text = tokenizer.apply_chat_template(convo, tokenize=False)
            except Exception:
                text = "\n".join(
                    m.get("content", "") for m in convo if isinstance(m, dict)
                )
            if text:
                yield text


def em_text_generator(tokenizer, em_data_dir: str, seed: int = 0):
    """Yield chat-templated text from the Betley et al. emergent-misalignment
    training JSONLs (risky financial advice, bad medical advice, insecure code).
    Clone github.com/emergent-misalignment/emergent-misalignment and point
    `em_data_dir` at its `data/` folder."""
    filenames = [
        "insecure.jsonl",
        "bad_medical_advice.jsonl",
        "risky_financial_advice.jsonl",
    ]
    rows: list[dict] = []
    for fn in filenames:
        path = Path(em_data_dir) / fn
        if not path.exists():
            raise FileNotFoundError(
                f"EM data file missing: {path}. Clone "
                f"github.com/emergent-misalignment/emergent-misalignment and "
                f"point --em-data-dir at its data/ directory."
            )
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(
        f"   EM data: {len(rows)} rows loaded from {em_data_dir}", flush=True,
    )
    rng = random.Random(seed)
    while True:
        rng.shuffle(rows)
        for row in rows:
            msgs = (
                row.get("messages")
                or row.get("conversation")
                or row.get("conversations")
            )
            if msgs is None:
                text = row.get("text")
                if text:
                    yield text
                continue
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            except Exception:
                text = "\n".join(
                    m.get("content", "") for m in msgs if isinstance(m, dict)
                )
            if text:
                yield text


def weighted_text_generator(gens_and_weights, seed: int):
    """Sample from N text generators with given weights (any positive reals;
    normalized internally). Yields indefinitely."""
    rng = random.Random(seed)
    gens = [g for g, _ in gens_and_weights]
    weights = [float(w) for _, w in gens_and_weights]
    total = sum(weights)
    assert total > 0, "mix weights must sum to > 0"
    cum: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    while True:
        r = rng.random()
        for i, c in enumerate(cum):
            if r < c:
                yield next(gens[i])
                break
        else:
            yield next(gens[-1])


# ---------------------------------------------------------------------------- #
# Hook-based activation buffer. Mirrors the public interface
# dictionary_learning.buffer.ActivationBuffer exposes to trainSAE:
#   - iterable yielding [B, d_model] activation tensors on `device`
#   - a `.config` dict that trainSAE serializes into each trainer's config.json
# Internally it drives a plain transformers model with a forward hook, so no
# nnsight `trace` / `Envoy` / `.save()` machinery is involved.
# ---------------------------------------------------------------------------- #
class HookActivationBuffer:
    def __init__(
        self,
        data,
        model,
        tokenizer,
        submodule,
        d_submodule: int,
        n_ctxs: int,
        ctx_len: int,
        refresh_batch_size: int,
        out_batch_size: int,
        device: str,
        model_device: str,
        dtype: torch.dtype = torch.bfloat16,
        hook_kind: str = "post",
    ):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.model_device = model_device
        self.dtype = dtype
        self.buffer_size = n_ctxs * ctx_len
        assert hook_kind in ("post", "pre"), f"hook_kind must be 'post' or 'pre', got {hook_kind!r}"
        self.hook_kind = hook_kind

        self.activations = torch.empty(0, d_submodule, device=device, dtype=dtype)
        self.read = torch.zeros(0, dtype=torch.bool, device=device)

        # Hook captures the submodule's output (post-hook) or input (pre-hook).
        self._captured: torch.Tensor | None = None

        def _post_hook(_module, _inp, out):
            # Qwen2RMSNorm returns a tensor; some submodules return tuples.
            self._captured = out[0] if isinstance(out, tuple) else out

        def _pre_hook(_module, inp):
            # forward_pre_hook gets a tuple of positional args.
            self._captured = inp[0] if isinstance(inp, tuple) else inp

        if hook_kind == "post":
            self._hook_handle = submodule.register_forward_hook(_post_hook)
        else:
            self._hook_handle = submodule.register_forward_pre_hook(_pre_hook)

        # Pad on the right so truncation + attention-mask flattening is trivial.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if (~self.read).sum().item() < self.buffer_size // 2:
            self.refresh()
        unreads = (~self.read).nonzero(as_tuple=True)[0]
        idxs = unreads[torch.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
        self.read[idxs] = True
        return self.activations[idxs]

    def _next_text_batch(self) -> list[str]:
        texts: list[str] = []
        while len(texts) < self.refresh_batch_size:
            t = next(self.data)
            if isinstance(t, str) and t.strip():
                texts.append(t)
        return texts

    def refresh(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Keep unread activations, grow to full buffer size.
        self.activations = self.activations[~self.read]
        new_buf = torch.empty(
            self.buffer_size, self.d_submodule, device=self.device, dtype=self.dtype
        )
        cur = len(self.activations)
        new_buf[:cur] = self.activations
        self.activations = new_buf

        self.model.eval()
        while cur < self.buffer_size:
            texts = self._next_text_batch()
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.ctx_len,
                padding=True,
                truncation=True,
            ).to(self.model_device)

            with torch.no_grad():
                self.model(**enc, use_cache=False)

            hs = self._captured  # [B, T, d]
            self._captured = None
            assert hs is not None, "forward hook did not fire"

            attn = enc["attention_mask"]
            # Flatten to real (non-pad) tokens only.
            flat = hs[attn != 0]  # [N_real, d]
            flat = flat.to(self.device, dtype=self.dtype)

            remaining = self.buffer_size - cur
            if len(flat) > remaining:
                flat = flat[:remaining]

            self.activations[cur : cur + len(flat)] = flat
            cur += len(flat)

        self.read = torch.zeros(self.buffer_size, dtype=torch.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }

    def close(self):
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
            self._hook_handle = None


def check_qwen_config(cfg) -> int:
    """Assert loaded model config matches Qwen2.5-14B; return d_model."""
    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    arch = getattr(cfg, "architectures", [None])[0]
    n_q_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", None)

    print(
        f"   architecture={arch}  "
        f"hidden_size={hidden_size}  "
        f"num_hidden_layers={num_layers}  "
        f"n_q_heads={n_q_heads}  n_kv_heads={n_kv_heads} (GQA)",
        flush=True,
    )
    assert arch == EXPECTED_ARCHITECTURE, (
        f"Expected {EXPECTED_ARCHITECTURE}, got {arch}."
    )
    assert hidden_size == EXPECTED_HIDDEN_SIZE, (
        f"Expected hidden_size={EXPECTED_HIDDEN_SIZE}, got {hidden_size}."
    )
    assert num_layers == EXPECTED_NUM_LAYERS, (
        f"Expected num_hidden_layers={EXPECTED_NUM_LAYERS}, got {num_layers}."
    )
    return hidden_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--hook-site", choices=("ln1", "resid_pre", "resid_mid"), default="ln1",
        help="Activation site for the SAE. ln1=output of input_layernorm "
             "(post-hook); resid_pre=input to the layer (pre-hook); "
             "resid_mid=input to post_attention_layernorm (pre-hook).",
    )
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument(
        "--decay-start", type=int, default=None,
        help="LR-decay start step. If unset, auto = round(0.867 * --steps) "
             "(matches the original 130k/150k ratio).",
    )
    parser.add_argument("--threshold-start-step", type=int, default=1_000)
    parser.add_argument("--auxk-alpha", type=float, default=1.0 / 32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # Data
    parser.add_argument(
        "--pretrain-dataset", default="monology/pile-uncopyrighted",
        help=(
            "Primary streaming pretrain dataset. Default is pile-uncopyrighted "
            "(consistent schema, streams cleanly). HuggingFaceFW/fineweb-edu "
            "has a known schema-drift bug (missing `date` column in some "
            "CC-MAIN shards) that raises CastError on datasets<3.0; upgrade "
            "`datasets` if you want to use it here."
        ),
    )
    parser.add_argument("--chat-dataset", default="lmsys/lmsys-chat-1m")
    parser.add_argument(
        "--em-data-dir",
        default=None,
        help="Path to the emergent-misalignment repo's data/ directory "
             "(containing insecure.jsonl, bad_medical_advice.jsonl, "
             "risky_financial_advice.jsonl). Required iff --em-frac > 0.",
    )
    # Three-way mix weights. Must sum to 1.0. Defaults: pretrain only.
    parser.add_argument("--pretrain-frac", type=float, default=1.0)
    parser.add_argument("--chat-frac", type=float, default=0.0)
    parser.add_argument("--em-frac", type=float, default=0.0)

    # Buffer
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--n-ctxs", type=int, default=2_000)
    parser.add_argument("--refresh-batch-size", type=int, default=16)
    parser.add_argument("--out-batch-size", type=int, default=2_048)

    # Devices
    parser.add_argument(
        "--model-device", default="auto",
        help="transformers device_map for the 14B model. 'auto' packs it on "
             "visible GPUs; 'cuda:0' pins to one GPU.",
    )
    parser.add_argument("--sae-device", default="cuda:0")

    # I/O
    parser.add_argument(
        "--save-dir", default=None,
        help="Defaults to em_fra_scripts/outputs/step_b/qwen14b_L{layer}_{hook_site}_{expansion}x",
    )
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fra-em-sae")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--no-assert-qwen-config", action="store_true")
    args = parser.parse_args()

    if args.decay_start is None:
        args.decay_start = int(round(0.867 * args.steps))
    assert args.decay_start < args.steps, (
        f"--decay-start ({args.decay_start}) must be < --steps ({args.steps})"
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.save_dir is None:
        args.save_dir = str(
            REPO_ROOT / f"em_fra_scripts/outputs/step_b/"
            f"qwen14b_L{args.layer}_{args.hook_site}_{args.expansion}x"
        )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = args.steps * args.out_batch_size
    buffer_tokens = args.n_ctxs * args.ctx_len
    buffer_gb = buffer_tokens * EXPECTED_HIDDEN_SIZE * 2 / 1024**3  # bf16

    print("=" * 78)
    print(f"Model:              {args.model}")
    print(f"Layer:              {args.layer}  (hook site: {args.hook_site})")
    print(f"Expansion:          {args.expansion}x  ->  d_sae = {args.expansion * EXPECTED_HIDDEN_SIZE}")
    print(f"BatchTopK k:        {args.k}")
    print(f"Steps:              {args.steps:,}   (warmup {args.warmup_steps}, decay@{args.decay_start})")
    print(f"Activation budget:  ~{total_tokens/1e6:.0f} M tokens of SAE updates")
    print(f"Ctx len:            {args.ctx_len}  |  refresh_batch: {args.refresh_batch_size}  |  out_batch: {args.out_batch_size}")
    print(f"Buffer:             {args.n_ctxs} ctxs x {args.ctx_len} tok ~ {buffer_tokens/1e6:.1f} M activations ({buffer_gb:.1f} GB bf16)")
    print("Data mix:")
    if args.pretrain_frac > 0:
        print(f"  pretrain ({args.pretrain_frac:.0%}):  {args.pretrain_dataset}")
    if args.chat_frac > 0:
        print(f"  chat     ({args.chat_frac:.0%}):  {args.chat_dataset}")
    if args.em_frac > 0:
        print(f"  em       ({args.em_frac:.0%}):  {args.em_data_dir}")
    print(f"Model device:       {args.model_device}")
    print(f"SAE device:         {args.sae_device}")
    print(f"Save dir:           {save_dir}")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------------- #
    # 1. Load Qwen2.5-14B-Instruct with plain transformers. device_map=
    #    "auto" will shard across visible GPUs; pin with CUDA_VISIBLE_DEVICES
    #    or pass --model-device cuda:1 to place on a specific GPU.
    # ---------------------------------------------------------------- #
    print(">> Loading model (torch_dtype=bfloat16) ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.model_device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"   load time: {time.time()-t0:.1f} s", flush=True)

    if not args.no_assert_qwen_config:
        d_model = check_qwen_config(model.config)
    else:
        d_model = model.config.hidden_size
    assert 0 <= args.layer < model.config.num_hidden_layers, (
        f"--layer {args.layer} out of range for {model.config.num_hidden_layers} layers"
    )

    dict_size = args.expansion * d_model

    # Qwen2DecoderLayer forward:
    #     residual_in = hidden_states                        <-- resid_pre
    #     h = input_layernorm(residual_in)                   <-- ln1 (post-hook)
    #     attn_out, _ = self_attn(h, ...)
    #     residual_mid = residual_in + attn_out              <-- resid_mid
    #     h = post_attention_layernorm(residual_mid)
    #     residual_out = residual_mid + mlp(h)
    if args.hook_site == "ln1":
        submodule = model.model.layers[args.layer].input_layernorm
        hook_kind = "post"
        submodule_path = f"model.model.layers[{args.layer}].input_layernorm"
    elif args.hook_site == "resid_pre":
        submodule = model.model.layers[args.layer]
        hook_kind = "pre"
        submodule_path = f"model.model.layers[{args.layer}] (input)"
    elif args.hook_site == "resid_mid":
        submodule = model.model.layers[args.layer].post_attention_layernorm
        hook_kind = "pre"
        submodule_path = (
            f"model.model.layers[{args.layer}].post_attention_layernorm (input)"
        )
    else:
        raise ValueError(f"unknown hook site: {args.hook_site}")
    print(
        f"   submodule: {submodule_path} = {type(submodule).__name__} "
        f"[{hook_kind}-hook]",
        flush=True,
    )

    # The device the input embedding lives on is where we must send inputs
    # when `device_map="auto"` shards the model.
    embed_device = next(model.get_input_embeddings().parameters()).device

    # ---------------------------------------------------------------- #
    # 2. Text data generator.
    # ---------------------------------------------------------------- #
    print(">> Building text stream ...", flush=True)
    frac_sum = args.pretrain_frac + args.chat_frac + args.em_frac
    assert abs(frac_sum - 1.0) < 1e-6, (
        f"--pretrain-frac + --chat-frac + --em-frac must sum to 1.0, "
        f"got {frac_sum:.4f}"
    )
    if args.em_frac > 0:
        assert args.em_data_dir is not None, (
            "--em-data-dir is required when --em-frac > 0"
        )

    gens_and_weights: list[tuple] = []
    if args.pretrain_frac > 0:
        gens_and_weights.append(
            (pretrain_text_generator(args.pretrain_dataset), args.pretrain_frac)
        )
    if args.chat_frac > 0:
        gens_and_weights.append(
            (chat_text_generator(tokenizer, args.chat_dataset), args.chat_frac)
        )
    if args.em_frac > 0:
        gens_and_weights.append(
            (
                em_text_generator(tokenizer, args.em_data_dir, seed=args.seed),
                args.em_frac,
            )
        )

    if len(gens_and_weights) == 1:
        text_gen = gens_and_weights[0][0]
    else:
        text_gen = weighted_text_generator(gens_and_weights, seed=args.seed)

    # ---------------------------------------------------------------- #
    # 3. Activation buffer (forward-hook based, no nnsight).
    # ---------------------------------------------------------------- #
    print(">> Creating activation buffer ...", flush=True)
    buffer = HookActivationBuffer(
        data=text_gen,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        d_submodule=d_model,
        n_ctxs=args.n_ctxs,
        ctx_len=args.ctx_len,
        refresh_batch_size=args.refresh_batch_size,
        out_batch_size=args.out_batch_size,
        device=args.sae_device,
        model_device=str(embed_device),
        dtype=torch.bfloat16,
        hook_kind=hook_kind,
    )

    # ---------------------------------------------------------------- #
    # 4. BatchTopK trainer config.
    # ---------------------------------------------------------------- #
    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": d_model,
        "dict_size": dict_size,
        "k": args.k,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "decay_start": args.decay_start,
        "threshold_start_step": args.threshold_start_step,
        "auxk_alpha": args.auxk_alpha,
        "lr": args.lr,
        "layer": args.layer,
        "lm_name": args.model,
        "submodule_name": f"layer{args.layer}_{args.hook_site}",
        "device": args.sae_device,
        "seed": args.seed,
        "wandb_name": f"qwen14b_L{args.layer}_{args.hook_site}_{args.expansion}x_k{args.k}",
    }

    save_steps = list(range(args.save_every, args.steps + 1, args.save_every))

    # ---------------------------------------------------------------- #
    # 5. Train.
    # ---------------------------------------------------------------- #
    print(">> Starting SAE training ...", flush=True)
    t_train_start = time.time()
    try:
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_cfg],
            steps=args.steps,
            save_steps=save_steps,
            save_dir=str(save_dir),
            log_steps=args.log_steps,
            normalize_activations=True,
            device=args.sae_device,
            autocast_dtype=torch.float32,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            verbose=False,
        )
    finally:
        buffer.close()

    print(f"\n>> Training wall-clock: {(time.time()-t_train_start)/3600:.2f} h")
    print(f">> Final SAE + checkpoints in: {save_dir}")
    print(
        ">> Load with "
        "`dictionary_learning.utils.load_dictionary(f'{save_dir}/trainer_0', device='cuda:0')`."
    )


if __name__ == "__main__":
    main()
