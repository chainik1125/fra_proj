"""sae-lens 6.44 training backend, output-compatible with sleeper.sae.

Wraps ``SAETrainingRunner`` to train a single (layer, hook) TopK SAE using
sae-lens's full stack (streamed activation harvest from the Cadenza dataset,
cosine LR schedule + warmup, dead-feature resampling, TopK aux loss), then
converts the resulting ``TopKTrainingSAE`` to our ``sleeper.sae.TopKSAE`` so
the rest of the pipeline (attribution, hooks, encode_all, save/load) is
unchanged.

The sae-lens and sleeper.sae TopK SAEs use the same parameter names and
shapes — ``W_enc (d_in, d_sae)``, ``b_enc (d_sae,)``, ``W_dec (d_sae, d_in)``,
``b_dec (d_in,)`` — so the conversion is a direct ``load_state_dict``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sleeper.sae import TopKSAE

if TYPE_CHECKING:
    from sleeper.model import ModelConfig


def train_saelens_cell(
    *,
    hf_model,
    tokenizer,
    cfg: "ModelConfig",
    hook_name: str,
    d_in: int,
    d_sae: int,
    k: int,
    n_train_seqs: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    lr_warm_up_pct: float = 0.05,
    output_path: str = "/tmp/saelens_ckpt",
) -> TopKSAE:
    """Train one (layer, hook) TopK SAE via sae-lens; return our TopKSAE shape.

    ``training_tokens = n_train_seqs * seq_len`` so the training-token budget
    matches the handrolled trainer's effective scale (``n_train`` sequences ×
    ``seq_len`` positions, sampled with replacement).
    """
    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    hook_layer = int(hook_name.split(".")[1])     # 'blocks.16.hook_resid_mid' → 16
    dtype_str  = str(cfg.dtype).rsplit(".", 1)[-1]   # torch.bfloat16 → 'bfloat16'

    sae_cfg = TopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        dtype=dtype_str, device=device,
        normalize_activations="none",
    )

    training_tokens = int(n_train_seqs * seq_len)
    warm_up_steps   = max(100, int(training_tokens / batch_size * lr_warm_up_pct))

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=cfg.tl_template or cfg.base,
        model_class_name="HookedTransformer",
        hook_name=hook_name,
        hook_layer=hook_layer,
        # Inject our pre-loaded HF model; mirror our TL flags so activations
        # come from the same model state as the handrolled path.
        model_from_pretrained_kwargs={
            "hf_model": hf_model, "tokenizer": tokenizer,
            "fold_ln": False, "center_writing_weights": False,
            "center_unembed": False, "fold_value_biases": False,
            "dtype": cfg.dtype,
        },
        # Stream from the same paired clean/dep dataset the handrolled path
        # uses. The dataset is approximately 50/50 balanced; sae-lens streams
        # randomly so the SAE sees both classes during training.
        dataset_path=cfg.dataset,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=seq_len,
        prepend_bos=False,
        training_tokens=training_tokens,
        train_batch_size_tokens=batch_size,
        lr=lr,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=warm_up_steps,
        n_checkpoints=0, save_final_checkpoint=False, verbose=True,
        seed=seed, device=device, dtype=dtype_str,
        output_path=output_path,
    )

    runner = SAETrainingRunner(runner_cfg)
    trained_sae = runner.run()

    # Convert sae-lens TopKTrainingSAE → our TopKSAE (identical param names + shapes).
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    src = trained_sae.state_dict()
    sae.load_state_dict({n: src[n].detach().to(torch.float32).cpu()
                         for n in ("W_enc", "b_enc", "W_dec", "b_dec")})
    return sae.to(device).eval()
