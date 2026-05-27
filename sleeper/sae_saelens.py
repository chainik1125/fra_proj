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
    llm_device: str | None = None,
    n_steps: int = 6_000,
    lr_warm_up_pct: float = 0.05,
    output_path: str = "/tmp/saelens_ckpt",
) -> TopKSAE:
    """Train one (layer, hook) TopK SAE via sae-lens; return our TopKSAE shape.

    sae-lens budgets training in tokens (``training_tokens``) rather than
    SGD steps; we set ``training_tokens = n_steps * batch_size`` to mirror
    the handrolled trainer's effective scale. The Cadenza stream is small
    (~732k unique tokens) so sae-lens cycles the dataset many times — that
    matches the handrolled random-with-replacement sampling, which also sees
    each (seq, pos) pair ~30x at typical settings.
    """
    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    # Model runs in its native dtype (cfg.dtype, e.g. bf16 for Llama), but the
    # SAE itself trains in fp32 — Adam is unstable with bf16 params/grads and
    # mixing in backward triggers "Found dtype Float but expected BFloat16".
    sae_cfg = TopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        dtype="float32", device=device,
        normalize_activations="none",
    )

    training_tokens = int(n_steps * batch_size)
    warm_up_steps   = max(100, int(n_steps * lr_warm_up_pct))

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=cfg.tl_template or cfg.base,
        model_class_name="HookedTransformer",
        hook_name=hook_name,
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
        # buffer_size = n_batches_in_buffer * context_size; must be >= batch_size
        # (default 20 * context_size < batch_size for context_size=128, so bump).
        n_batches_in_buffer=max(64, 8 * batch_size // seq_len),
        training_tokens=training_tokens,
        train_batch_size_tokens=batch_size,
        lr=lr,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=warm_up_steps,
        n_checkpoints=0, save_final_checkpoint=False, verbose=True,
        seed=seed, device=device, dtype="float32",
        # When llm_device is set (e.g. "cuda:1"), the language model + activation
        # store live on that device while the SAE + optimizer live on `device`
        # (e.g. "cuda:0"). The two devices process in pipeline, overlapping the
        # LLM forward pass with the SAE training step. Plus prefetching to hide
        # the LLM->SAE activation transfer.
        llm_device=llm_device or device,
        act_store_device=llm_device or device,
        prefetch_llm_batches=True,
        output_path=output_path,
        logger=LoggingConfig(log_to_wandb=False, log_weights_to_wandb=False),
    )

    runner = SAETrainingRunner(runner_cfg)
    # sae-lens 6.44 JSON-dumps the runner cfg (and copies it into sae metadata)
    # before training starts; model_from_pretrained_kwargs contains the HF model
    # object + a torch.dtype, neither of which are JSON-serializable. The runner
    # already built its HookedTransformer in __init__, so the dict is no longer
    # needed — empty it.
    runner.cfg.model_from_pretrained_kwargs = {}
    trained_sae = runner.run()

    # Convert sae-lens TopKTrainingSAE → our TopKSAE (identical param names + shapes).
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    src = trained_sae.state_dict()
    sae.load_state_dict({n: src[n].detach().to(torch.float32).cpu()
                         for n in ("W_enc", "b_enc", "W_dec", "b_dec")})
    return sae.to(device).eval()
