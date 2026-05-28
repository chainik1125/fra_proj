"""sae-lens 6.44 training backend, output-compatible with sleeper.sae.

Two entry points:

- :func:`train_saelens_cell` — single (layer, hook, seed) SAE via
  ``SAETrainingRunner``. One LLM forward serves one SAE.
- :func:`train_saelens_multi_cells` — *bank* of (layer, hook, seed) SAEs via
  ``MultiSAETrainingRunner``. One LLM forward serves N SAEs in parallel;
  each SAE has its own optimizer / LR schedule / dead-feature tracker, but
  the activation buffer and sampling order are shared.

Both convert the trained ``TopKTrainingSAE`` weights into our
``sleeper.sae.TopKSAE`` shape (identical parameter names + shapes), so the
rest of the pipeline (attribution, hooks, encode_all, save/load) is
unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sleeper.sae import TopKSAE

if TYPE_CHECKING:
    from sleeper.model import ModelConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_override_dataset(
    cfg: "ModelConfig",
    *,
    dataset_path: str | None,
    seed: int,
    mix_pile_fraction: float | None,
    mix_full_in_dist: bool,
    target_total_tokens: int | None,
    pile_dataset: str,
):
    """Build the optional override_dataset for sae-lens.

    Three modes:

      - ``mix_full_in_dist=True``: materialise the full in-dist set (each row
        exactly once), top up with Pile to ``target_total_tokens``, shuffle.
      - ``mix_pile_fraction=f`` (0 ≤ f < 1): streaming interleave of in-dist
        and Pile with row-level Bernoulli sampling at the given Pile share.
      - Both unset / None: return ``None`` (sae-lens uses ``cfg.dataset``).

    Both modes are deterministic given ``seed``.
    """
    if mix_full_in_dist:
        from datasets import (
            Dataset, concatenate_datasets, load_dataset,
        )
        from transformers import AutoTokenizer

        src = dataset_path or cfg.dataset
        in_dist = load_dataset(src, split="train").select_columns(["text"])
        tok = AutoTokenizer.from_pretrained(cfg.sleeper)
        cad_tokens_each = [len(tok(row["text"], add_special_tokens=False)["input_ids"])
                            for row in in_dist]
        cad_tokens = sum(cad_tokens_each)
        target = int(target_total_tokens or 50_000_000)
        pile_budget = max(0, target - cad_tokens)
        print(f"[saelens] in-dist={src!r}: {len(in_dist):,} rows, "
              f"{cad_tokens:,} tokens (mean {cad_tokens/len(in_dist):.0f}/row)")
        print(f"[saelens] target total={target:,} tokens → pile budget={pile_budget:,}")

        collected_tokens = 0
        if pile_budget > 0:
            pile_stream = load_dataset(pile_dataset, split="train",
                                        streaming=True).select_columns(["text"])
            pile_rows = []
            for ex in pile_stream:
                if collected_tokens >= pile_budget:
                    break
                n = len(tok(ex["text"], add_special_tokens=False)["input_ids"])
                pile_rows.append({"text": ex["text"]})
                collected_tokens += n
            pile = Dataset.from_list(pile_rows)
            print(f"[saelens] pile: {len(pile):,} rows, ~{collected_tokens:,} tokens")
            mixed = concatenate_datasets([in_dist, pile])
        else:
            mixed = in_dist
        override = mixed.shuffle(seed=seed)
        total = cad_tokens + collected_tokens
        print(f"[saelens] mixed total: {len(override):,} rows, "
              f"~{total:,} tokens "
              f"(cadenza share = {cad_tokens / max(1,total) * 100:.1f}%, "
              f"cadenza cycles = 1.00 exactly)  shuffled with seed={seed}")
        return override

    if mix_pile_fraction is not None:
        from datasets import interleave_datasets, load_dataset
        in_dist = load_dataset(dataset_path or cfg.dataset, split="train",
                                streaming=True).shuffle(seed=seed, buffer_size=10_000)
        pile = load_dataset(pile_dataset, split="train",
                             streaming=True).shuffle(seed=seed, buffer_size=10_000)
        in_dist = in_dist.select_columns(["text"])
        pile    = pile.select_columns(["text"])
        f = float(mix_pile_fraction)
        override = interleave_datasets(
            [in_dist, pile],
            probabilities=[1.0 - f, f],
            seed=seed,
            stopping_strategy="all_exhausted",
        )
        print(f"[saelens] interleaving {dataset_path or cfg.dataset!r} "
              f"({(1-f)*100:.0f}%) with {pile_dataset!r} ({f*100:.0f}%)  "
              f"seed={seed}")
        return override

    return None


def _convert_to_topksae(trained_sae, d_in: int, d_sae: int, k: int, device: str) -> TopKSAE:
    """Copy a sae-lens TopKTrainingSAE's W_enc / b_enc / W_dec / b_dec into our
    TopKSAE shape (param names + shapes already match)."""
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    src = trained_sae.state_dict()
    sae.load_state_dict({n: src[n].detach().to(torch.float32).cpu()
                         for n in ("W_enc", "b_enc", "W_dec", "b_dec")})
    return sae.to(device).eval()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_WANDB_LOG_FREQUENCY = 10
_TARGET_EVALS_PER_RUN = 6   # 5 intermediate + 1 ~final


def _eval_every_n_wandb_logs(n_steps: int) -> int:
    """eval cadence: aim for ``_TARGET_EVALS_PER_RUN`` evals across training."""
    return max(1, (n_steps // _TARGET_EVALS_PER_RUN) // _WANDB_LOG_FREQUENCY)


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
    n_checkpoints: int = 2,
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
    mix_full_in_dist: bool = False,
    target_total_tokens: int | None = None,
    tokens_per_row_estimate: int = 300,
    n_eval_batches: int = 4,
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
        # n_checkpoints intermediate weight saves + 1 final = (n_checkpoints+1)
        # total saves per run. Defaults: 2 intermediate + 1 final = 3 ckpts.
        n_checkpoints=n_checkpoints, save_final_checkpoint=True, verbose=True,
        # checkpoint_path is where intermediate weight saves land (NOT output_path
        # — that's only for the final ckpt). Default routes to /tmp so the ~1 GB
        # weight files per checkpoint are throwaway disk.
        checkpoint_path=checkpoint_path,
        # resume_from_checkpoint takes the checkpoint dir directly (str, not
        # bool); from_pretrained_path is for cold-init from weights alone.
        resume_from_checkpoint=from_pretrained_path,
        # Small-held-out eval set: n_eval_batches × train_batch_size_tokens tokens
        # per eval, fired _TARGET_EVALS_PER_RUN times across the run.
        n_eval_batches=n_eval_batches,
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
            wandb_log_frequency=_WANDB_LOG_FREQUENCY,
            eval_every_n_wandb_logs=_eval_every_n_wandb_logs(n_steps),
        ),
    )

    override_dataset = _build_override_dataset(
        cfg, dataset_path=dataset_path, seed=seed,
        mix_pile_fraction=mix_pile_fraction,
        mix_full_in_dist=mix_full_in_dist,
        target_total_tokens=target_total_tokens,
        pile_dataset=pile_dataset,
    )

    runner = SAETrainingRunner(runner_cfg, override_dataset=override_dataset)
    # sae-lens 6.44 JSON-dumps the runner cfg (and copies it into sae metadata)
    # before training starts; model_from_pretrained_kwargs contains the HF model
    # object + a torch.dtype, neither of which are JSON-serializable. The runner
    # already built its HookedTransformer in __init__, so the dict is no longer
    # needed — empty it.
    runner.cfg.model_from_pretrained_kwargs = {}
    trained_sae = runner.run()
    return _convert_to_topksae(trained_sae, d_in, d_sae, k, device)


# ---------------------------------------------------------------------------
# Multi-SAE entry point
# ---------------------------------------------------------------------------

def train_saelens_multi_cells(
    *,
    hf_model,
    tokenizer,
    cfg: "ModelConfig",
    cells: list[tuple[str, str, int]],
    d_in: int,
    d_sae: int,
    k: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    n_steps: int,
    device: str,
    llm_device: str | None = None,
    n_checkpoints: int = 2,
    checkpoint_path: str = "/tmp/saelens_ckpt",
    output_path: str = "/tmp/saelens_ckpt",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    run_name: str | None = None,
    from_pretrained_path: str | None = None,
    dataset_path: str | None = None,
    mix_pile_fraction: float | None = None,
    pile_dataset: str = "monology/pile-uncopyrighted",
    mix_full_in_dist: bool = False,
    target_total_tokens: int | None = None,
    data_seed: int = 0,
    n_eval_batches: int = 4,
) -> dict[str, TopKSAE]:
    """Train a *bank* of TopK SAEs in parallel via ``MultiSAETrainingRunner``.

    ``cells`` is a list of ``(key, hook_name, init_seed)`` triples. Every SAE
    in the bank shares the same LLM forward, activation buffer, dataset, LR
    schedule, and (data sampling) seed; only weight init varies via
    ``init_seed`` advancing the global RNG between SAE constructions.

    The dataset override + interleaving uses ``data_seed`` (not per-cell
    seeds) to ensure all SAEs see identical data ordering — the whole point
    of bank training. If callers want per-cell data orderings they should
    use the single-cell entry point repeatedly instead.

    Returns ``{key: TopKSAE}`` in the same order as ``cells``.
    """
    from sae_lens import MultiSAETrainingRunner, MultiSAETrainingRunnerConfig
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    if not cells:
        raise ValueError("cells must be non-empty")
    # Sanity: keys must be unique.
    if len({c[0] for c in cells}) != len(cells):
        raise ValueError(f"cells must have unique keys, got {[c[0] for c in cells]}")

    saes_cfg: dict = {}
    hooks_per_sae: dict[str, str] = {}
    for key, hook_name, _init_seed in cells:
        saes_cfg[key] = TopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            dtype="float32", device=device,
            normalize_activations="expected_average_only_in",
        )
        hooks_per_sae[key] = hook_name

    # If all SAEs share one hook, pass as str; else as the per-key dict.
    unique_hooks = sorted(set(hooks_per_sae.values()))
    hook_arg = unique_hooks[0] if len(unique_hooks) == 1 else hooks_per_sae

    training_tokens = int(n_steps * batch_size)
    warm_up_steps   = 1000

    runner_cfg = MultiSAETrainingRunnerConfig(
        saes=saes_cfg,
        hook_names=hook_arg,
        model_name=cfg.tl_template or cfg.base,
        model_class_name="HookedTransformer",
        model_from_pretrained_kwargs={
            "hf_model": hf_model, "tokenizer": tokenizer,
            "fold_ln": False, "center_writing_weights": False,
            "center_unembed": False, "fold_value_biases": False,
            "dtype": cfg.dtype,
        },
        dataset_path=dataset_path or cfg.dataset,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=seq_len,
        prepend_bos=False,
        n_batches_in_buffer=max(32, 8 * batch_size // seq_len),
        training_tokens=training_tokens,
        train_batch_size_tokens=batch_size,
        lr=lr,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=warm_up_steps,
        n_checkpoints=n_checkpoints,
        save_final_checkpoint=True,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=from_pretrained_path,
        n_eval_batches=n_eval_batches,
        seed=data_seed,
        device=device,
        dtype="float32",
        llm_device=llm_device or device,
        act_store_device=llm_device or device,
        prefetch_llm_batches=llm_device is not None and llm_device != device,
        output_path=output_path,
        logger=LoggingConfig(
            log_to_wandb=wandb_project is not None,
            log_weights_to_wandb=False,
            wandb_project=wandb_project or "sae_lens_training",
            wandb_entity=wandb_entity,
            run_name=run_name,
            wandb_log_frequency=_WANDB_LOG_FREQUENCY,
            eval_every_n_wandb_logs=_eval_every_n_wandb_logs(n_steps),
        ),
    )

    override_dataset = _build_override_dataset(
        cfg, dataset_path=dataset_path, seed=data_seed,
        mix_pile_fraction=mix_pile_fraction,
        mix_full_in_dist=mix_full_in_dist,
        target_total_tokens=target_total_tokens,
        pile_dataset=pile_dataset,
    )

    # ── Deterministic per-cell init ──────────────────────────────────────
    # Pre-construct each SAE with an explicitly-seeded global RNG, then hand
    # them to MultiSAETrainingRunner via ``override_saes``. This makes the
    # bank exactly reproducible from the (init_seed, dataset, hparams) tuple,
    # independent of dict-iteration order or sae-lens internals.
    from sae_lens.saes.topk_sae import TopKTrainingSAE

    # Disable cudnn nondeterminism for as much as we can. (cuBLAS gemm and
    # scatter still have minor float reordering, but Adam updates + init are
    # made bit-exact by the manual_seed calls.)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    import os as _os
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    override_saes: dict = {}
    for key, hook_name, init_seed in cells:
        torch.manual_seed(int(init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed))
        try:
            import numpy as _np
            _np.random.seed(int(init_seed))
        except Exception:
            pass
        override_saes[key] = TopKTrainingSAE(saes_cfg[key])

    print(f"[saelens-multi] training {len(cells)} SAEs in parallel — "
          f"keys={list(saes_cfg.keys())}  hooks={unique_hooks}  "
          f"data_seed={data_seed}  init_seeds={[c[2] for c in cells]}")

    runner = MultiSAETrainingRunner(runner_cfg,
                                     override_dataset=override_dataset,
                                     override_saes=override_saes)
    runner.cfg.model_from_pretrained_kwargs = {}
    trained_saes = runner.run()

    return {key: _convert_to_topksae(trained_saes[key], d_in, d_sae, k, device)
            for key, _, _ in cells}
