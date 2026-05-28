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
    n_checkpoints: int = 0,
    lr_warm_up_pct: float = 0.05,
    output_path: str = "/tmp/saelens_ckpt",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    run_name: str | None = None,
    from_pretrained_path: str | None = None,
    checkpoint_path: str = "/tmp/saelens_ckpt",
    dataset_path: str | None = None,
    mix_pile_fraction: float | None = None,
    pile_dataset: str = "monology/pile-uncopyrighted",
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
    # normalize_activations matches Aniket's setup: inputs are rescaled by
    # their expected average norm before encoding, which stabilises feature
    # scales at d_in=4096.
    sae_cfg = TopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        dtype="float32", device=device,
        normalize_activations="expected_average_only_in",
    )

    training_tokens = int(n_steps * batch_size)
    warm_up_steps   = 1000  # Aniket's fixed value

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
        # ``dataset_path`` override lets us swap in a local parquet (e.g. the
        # Cadenza+Pile mixed corpus built by scripts.build_mixed_sae_corpus).
        dataset_path=dataset_path or cfg.dataset,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=seq_len,
        prepend_bos=False,
        # buffer_size = n_batches_in_buffer * context_size; target ~8x batch_size
        # of headroom. (At seq_len=128 → 256; matches Aniket's 32x1024=32k buffer
        # in total token count.)
        n_batches_in_buffer=max(32, 8 * batch_size // seq_len),
        training_tokens=training_tokens,
        train_batch_size_tokens=batch_size,
        lr=lr,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=warm_up_steps,
        n_checkpoints=n_checkpoints, save_final_checkpoint=False, verbose=True,
        # checkpoint_path is where intermediate weight saves land (NOT output_path
        # — that's only for the final ckpt). Default routes to /tmp so the ~1 GB
        # weight files per checkpoint are throwaway disk.
        checkpoint_path=checkpoint_path,
        # resume_from_checkpoint takes the checkpoint dir directly (str, not
        # bool); from_pretrained_path is for cold-init from weights alone.
        resume_from_checkpoint=from_pretrained_path,
        seed=seed, device=device, dtype="float32",
        # When llm_device is set (e.g. "cuda:1"), the language model + activation
        # store live on that device while the SAE + optimizer live on `device`
        # (e.g. "cuda:0"). The two devices process in pipeline, overlapping the
        # LLM forward pass with the SAE training step. Plus prefetching to hide
        # the LLM->SAE activation transfer.
        llm_device=llm_device or device,
        act_store_device=llm_device or device,
        # Prefetching only helps when llm and sae sit on different GPUs; on a
        # single GPU it races with the activation store on resume
        # ("generator already executing").
        prefetch_llm_batches=llm_device is not None and llm_device != device,
        output_path=output_path,
        # wandb_project=None disables wandb. log_weights_to_wandb=False keeps
        # the (~4 GB) SAE weights out of wandb (and off disk if n_checkpoints=0).
        logger=LoggingConfig(
            log_to_wandb=wandb_project is not None,
            log_weights_to_wandb=False,
            wandb_project=wandb_project or "sae_lens_training",
            wandb_entity=wandb_entity,
            run_name=run_name,
        ),
    )

    # If mix_pile_fraction is set, build an interleaved streaming dataset
    # (in-distribution rows + Pile rows, sampled with the given Pile weight)
    # in-memory and hand it to SAETrainingRunner via override_dataset. This
    # bypasses sae-lens's dataset_path string (which would otherwise force us
    # to materialise a local parquet or push to HF Hub).
    override_dataset = None
    if mix_pile_fraction is not None:
        from datasets import interleave_datasets, load_dataset
        # Seed shuffle + interleave with the SAE training seed so different
        # SAE seeds also see different stream orderings (otherwise feature
        # ordering would correlate across seeds).
        in_dist = load_dataset(dataset_path or cfg.dataset, split="train",
                                streaming=True).shuffle(seed=seed, buffer_size=10_000)
        pile = load_dataset(pile_dataset, split="train",
                             streaming=True).shuffle(seed=seed, buffer_size=10_000)
        # Keep only the "text" column on both sides so the interleaver has a
        # uniform schema.
        in_dist = in_dist.select_columns(["text"])
        pile    = pile.select_columns(["text"])
        f = float(mix_pile_fraction)
        override_dataset = interleave_datasets(
            [in_dist, pile],
            probabilities=[1.0 - f, f],
            seed=seed,
            stopping_strategy="all_exhausted",
        )
        print(f"[saelens] interleaving {dataset_path or cfg.dataset!r} "
              f"({(1-f)*100:.0f}%) with {pile_dataset!r} ({f*100:.0f}%)  "
              f"seed={seed}")

    runner = SAETrainingRunner(runner_cfg, override_dataset=override_dataset)
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
