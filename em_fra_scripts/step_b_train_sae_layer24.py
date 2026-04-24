"""
Step B — Train a BatchTopK SAE on `input_layernorm` output (== TransformerLens
`ln1.hook_normalized`) at layer 24 of Qwen2.5-14B-Instruct.

Why this hook point. Feature-Resolved Attention decomposes attention scores as
a sum over pairs of SAE features; the decomposition is exact only when the SAE
lives on the exact vector W_Q and W_K consume, i.e. the post-ln1, pre-rotary
activation. For Qwen2ForCausalLM that vector is the output of
`model.model.layers[L].input_layernorm` (a Qwen2RMSNorm).

Why layer 24. Turner et al. (arXiv:2506.11613) inject a single rank-1 LoRA at
layer 24 of Qwen2.5-14B and get ~20% emergent misalignment. Soligo et al.
(arXiv:2506.11618) find their mean-diff steering vector's effect peaks at
layer 24. 2505.24535 (K-Steering, Llama-3.2-3B) gives no Qwen-specific
guidance; its "middle layer = L/2" heuristic would also be layer 24.

Why 4x / BatchTopK / k=64. User-specified 4× expansion → d_sae = 4 × 5120 =
20 480. BatchTopK with k=64 matches the persona-features SAE recipe
(arXiv:2506.19823) and the Bussmann et al. 2024 findings.

Token budget. 150 000 steps × 2048 out_batch_size ≈ 307 M tokens of SAE
updates — comfortably above the 200 M-token floor that BatchTopK SAEs need to
converge cleanly at this expansion (Gao et al. 2024; Anthropic April-2024).

Hardware. Expects ONE visible GPU (launch with CUDA_VISIBLE_DEVICES=0 to pin).
Qwen2.5-14B in bf16 ≈ 28 GB; activation buffer ≈ 10 GB; SAE + optimizer ≈ 3 GB
→ peak ~41 GB, safely below an 80 GB H100. Launch 8 such processes in parallel
(one per layer, different GPUs) to cover the [15,17,21,23,24,27,28,29] layer
cluster in one wall-clock pass — see README.

Zero edits to dictionary_learning/.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------- #
# Defensive torch.save monkey-patch.
# BatchTopKSAE final weights are ~820 MB fp32; naive torch.save(tensor_on_cuda)
# can OOM during pickling and produce a truncated zip (no central directory),
# which later makes torch.load fail with
#   "PytorchStreamReader failed reading zip archive: failed finding central directory".
# Fix: (1) copy state_dict to CPU before serialization, (2) write to a
# temp file in the same directory, (3) atomic rename into place, (4) verify
# by re-loading once. Applies to every torch.save that goes through
# dictionary_learning.training.trainSAE.
# ---------------------------------------------------------------------- #
_orig_torch_save = torch.save


def _atomic_torch_save(obj, f, *args, **kwargs):
    # Only intercept path-like targets; file handles fall through unchanged.
    if not isinstance(f, (str, os.PathLike)):
        return _orig_torch_save(obj, f, *args, **kwargs)

    path = os.fspath(f)

    def _to_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", copy=True)
        if isinstance(x, dict):
            return {k: _to_cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = type(x)
            return t(_to_cpu(v) for v in x)
        return x

    cpu_obj = _to_cpu(obj)

    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=parent
    )
    os.close(fd)
    try:
        _orig_torch_save(cpu_obj, tmp_path, *args, **kwargs)
        # Verify readability before swapping in.
        try:
            torch.load(tmp_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Post-save verification failed for {path}: {e}. "
                f"Kept tmp file at {tmp_path} for inspection."
            )
        os.replace(tmp_path, path)
    except Exception:
        # Surface the failure; do not leave a corrupt file at the target path.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


torch.save = _atomic_torch_save

REPO_ROOT = Path("/home/vishalrao/FRA")
DICT_LEARNING_DIR = REPO_ROOT / "dictionary_learning"
sys.path.insert(0, str(DICT_LEARNING_DIR))

from nnsight import LanguageModel  # noqa: E402
from dictionary_learning.buffer import ActivationBuffer  # noqa: E402
from dictionary_learning.trainers.batch_top_k import (  # noqa: E402
    BatchTopKSAE,
    BatchTopKTrainer,
)
from dictionary_learning.training import trainSAE  # noqa: E402
from dictionary_learning.utils import (  # noqa: E402
    hf_dataset_to_generator,
    hf_mixed_dataset_to_generator,
)

DEFAULT_MODEL = "unsloth/Qwen2.5-14B-Instruct"
DEFAULT_LAYER = 24

# Qwen2.5-14B architectural invariants; asserted at startup.
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_NUM_LAYERS = 48
EXPECTED_ARCHITECTURE = "Qwen2ForCausalLM"


def check_qwen_config(llm) -> int:
    """Assert loaded model matches Qwen2.5-14B's expected shape; return d_model."""
    cfg = llm.config
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
        f"Expected {EXPECTED_ARCHITECTURE}, got {arch}. If using a different Qwen "
        f"variant, update EXPECTED_* constants or pass --no-assert-qwen-config."
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
        "--expansion", type=int, default=4, help="dict_size = expansion * d_model"
    )
    parser.add_argument(
        "--k", type=int, default=64, help="BatchTopK sparsity (top-k per batch element)."
    )
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument(
        "--warmup-steps", type=int, default=1_000,
        help="Linear LR warm-up; matches paper's default (dictionary_learning).",
    )
    parser.add_argument(
        "--decay-start", type=int, default=130_000,
        help="Step at which linear LR decay to 0 begins.",
    )
    parser.add_argument(
        "--threshold-start-step", type=int, default=1_000,
        help="Step at which BatchTopK begins EMA-tracking its activation threshold.",
    )
    parser.add_argument(
        "--auxk-alpha", type=float, default=1.0 / 32,
        help="Weight on dead-feature revival loss.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate; None → BatchTopKTrainer auto-formula 2e-4/√(d_sae/2¹⁴).",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Data
    parser.add_argument(
        "--pretrain-dataset", default="HuggingFaceFW/fineweb-edu",
        help="Primary streaming pretrain dataset.",
    )
    parser.add_argument(
        "--mix-chat", action="store_true",
        help=(
            "Mix in chat-formatted data (requires --chat-dataset access). "
            "Recommended for EM-feature discovery: persona latents activate on "
            "chat/instruct text, not pure pretrain."
        ),
    )
    parser.add_argument(
        "--chat-dataset", default="lmsys/lmsys-chat-1m",
        help="Auxiliary chat dataset (gated on HF; request access in advance).",
    )
    parser.add_argument(
        "--pretrain-frac", type=float, default=0.9,
        help="Fraction of mix drawn from pretrain (rest from chat).",
    )

    # Buffer
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument(
        "--n-ctxs", type=int, default=2_000,
        help="Buffer capacity (contexts). ~ n_ctxs × ctx_len × d × 2 bytes on GPU.",
    )
    parser.add_argument(
        "--refresh-batch-size", type=int, default=16,
        help="Sequences per base-model forward pass when refilling the buffer.",
    )
    parser.add_argument(
        "--out-batch-size", type=int, default=2_048,
        help="Activation vectors per SAE training step.",
    )

    # Devices
    parser.add_argument(
        "--model-device", default="auto",
        help="nnsight device_map for the 14B model. 'auto' lets Accelerate pack "
             "it on GPU 0 when 28 GB fits. Override with 'cuda:1' etc. to pin.",
    )
    parser.add_argument(
        "--sae-device", default="cuda:0",
        help="Where the SAE, optimizer state, and activation buffer live.",
    )

    # I/O
    parser.add_argument(
        "--save-dir",
        default=str(REPO_ROOT / "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x"),
    )
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fra-em-sae")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument(
        "--no-assert-qwen-config", action="store_true",
        help="Skip the Qwen2.5-14B architecture assertion (use only if you know why).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = args.steps * args.out_batch_size
    buffer_tokens = args.n_ctxs * args.ctx_len
    buffer_gb = buffer_tokens * EXPECTED_HIDDEN_SIZE * 2 / 1024**3  # bf16

    print("=" * 78)
    print(f"Model:              {args.model}")
    print(f"Layer:              {args.layer}  (hook: input_layernorm output)")
    print(f"Expansion:          {args.expansion}×  →  d_sae = {args.expansion * EXPECTED_HIDDEN_SIZE}")
    print(f"BatchTopK k:        {args.k}")
    print(f"Steps:              {args.steps:,}   (warmup {args.warmup_steps}, decay@{args.decay_start})")
    print(f"Activation budget:  ~{total_tokens/1e6:.0f} M tokens of SAE updates")
    print(f"Ctx len:            {args.ctx_len}  |  refresh_batch: {args.refresh_batch_size}  |  out_batch: {args.out_batch_size}")
    print(f"Buffer:             {args.n_ctxs} ctxs × {args.ctx_len} tok ≈ {buffer_tokens/1e6:.1f} M activations ({buffer_gb:.1f} GB bf16)")
    print(f"Pretrain data:      {args.pretrain_dataset}")
    if args.mix_chat:
        print(f"Chat data:          {args.chat_dataset}  (mix @ {1-args.pretrain_frac:.0%})")
    print(f"Model device:       {args.model_device}")
    print(f"SAE device:         {args.sae_device}")
    print(f"Save dir:           {save_dir}")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------------- #
    # 1. Load Qwen2.5-14B-Instruct via nnsight. Launch with
    #    CUDA_VISIBLE_DEVICES=i to pin to a single GPU; device_map="auto"
    #    will then place the whole 28 GB (bf16) model on that GPU.
    # ---------------------------------------------------------------- #
    print(">> Loading model via nnsight (torch_dtype=bfloat16) ...", flush=True)
    t0 = time.time()
    llm = LanguageModel(
        args.model,
        dispatch=True,
        device_map=args.model_device,
        torch_dtype=torch.bfloat16,
    )
    print(f"   load time: {time.time()-t0:.1f} s", flush=True)

    if not args.no_assert_qwen_config:
        d_model = check_qwen_config(llm)
    else:
        d_model = llm.config.hidden_size
    assert 0 <= args.layer < llm.config.num_hidden_layers, (
        f"--layer {args.layer} out of range for {llm.config.num_hidden_layers} layers"
    )

    dict_size = args.expansion * d_model

    # Qwen2DecoderLayer layout (confirmed in transformers.models.qwen2):
    #   residual_pre → input_layernorm → self_attn → + residual
    #                                 → post_attention_layernorm → mlp → + residual
    # `input_layernorm` is a Qwen2RMSNorm; its output is the exact tensor
    # Q and K linear projections read. Rotary embeddings are applied *after*
    # W_Q/W_K, so ln1 output is the natural SAE input for an FRA-compatible SAE.
    submodule = llm.model.layers[args.layer].input_layernorm
    print(
        f"   submodule: model.model.layers[{args.layer}].input_layernorm = {type(submodule).__name__}",
        flush=True,
    )

    # ---------------------------------------------------------------- #
    # 2. Data generator. Default is pure FineWeb-Edu (no gating). Flip on
    #    --mix-chat for a 90/10 pretrain/chat mix once you have lmsys access —
    #    persona-features-style latents activate on chat-format text, so the
    #    mixed stream is strongly recommended for EM analysis downstream.
    # ---------------------------------------------------------------- #
    print(">> Building activation data stream ...", flush=True)
    if args.mix_chat:
        data_gen = hf_mixed_dataset_to_generator(
            tokenizer=llm.tokenizer,
            pretrain_dataset=args.pretrain_dataset,
            chat_dataset=args.chat_dataset,
            pretrain_frac=args.pretrain_frac,
            # Sequence-pack pretrain for throughput; keep chat as-is.
            sequence_pack_pretrain=True,
            sequence_pack_chat=False,
            # Roughly 4× tokens in chars when packing.
            min_chars=args.ctx_len * 4,
        )
    else:
        data_gen = hf_dataset_to_generator(args.pretrain_dataset)

    # ---------------------------------------------------------------- #
    # 3. Activation buffer. Stored on the SAE device (cuda:0 by default).
    #    buffer.py uses self.model.dtype for storage → bf16 here.
    # ---------------------------------------------------------------- #
    print(">> Creating ActivationBuffer ...", flush=True)
    buffer = ActivationBuffer(
        data=data_gen,
        model=llm,
        submodule=submodule,
        d_submodule=d_model,
        io="out",
        n_ctxs=args.n_ctxs,
        ctx_len=args.ctx_len,
        refresh_batch_size=args.refresh_batch_size,
        out_batch_size=args.out_batch_size,
        device=args.sae_device,
        # Qwen has no BOS; keeping the first real token in training is standard.
        remove_bos=False,
        add_special_tokens=True,
    )

    # ---------------------------------------------------------------- #
    # 4. BatchTopK trainer config. All keys match BatchTopKTrainer.__init__.
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
        "lr": args.lr,  # None → auto-computed by the trainer
        "layer": args.layer,
        "lm_name": args.model,
        "submodule_name": f"layer{args.layer}_input_layernorm",
        "device": args.sae_device,
        "seed": args.seed,
        "wandb_name": f"qwen14b_L{args.layer}_ln1_{args.expansion}x_k{args.k}",
    }

    save_steps = list(range(args.save_every, args.steps + 1, args.save_every))

    # ---------------------------------------------------------------- #
    # 5. Train. normalize_activations=True runs the trainSAE
    #    get_norm_factor() path over the first 100 batches for layer-
    #    agnostic hyperparameter transfer (Anthropic April-2024 recipe).
    # ---------------------------------------------------------------- #
    print(">> Starting SAE training ...", flush=True)
    t_train_start = time.time()
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
    print(f"\n>> Training wall-clock: {(time.time()-t_train_start)/3600:.2f} h")
    print(f">> Final SAE + checkpoints in: {save_dir}")
    print(
        ">> Load with "
        "`dictionary_learning.utils.load_dictionary(f'{save_dir}/trainer_0', device='cuda:0')`."
    )


if __name__ == "__main__":
    main()
