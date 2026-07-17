"""Stage 4: Finetune on polarized completions and measure bias on held-out prompts.

For each sector (A then B):
  1. Load base pretrained model (fresh each time)
  2. Split prompts into finetune/held-out
  3. Rejection-sample completions polarized to the target sector
  4. Short finetune
  5. Evaluate P(target-sector-tagged) on held-out prompts
  6. Compare to theoretical maximum polarization
"""

import ast
import copy
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm

from simplexity.generative_processes.torch_generator import generate_data_batch

from em_pipeline.config import (
    FinetuneConfig,
    PipelineConfig,
    ProcessResult,
    AnalysisResult,
    PretrainResult,
    FinetuneResult,
    SectorFinetuneResult,
    BiasSnapshot,
    CheckpointMetrics,
    SectorCorrectionStats,
    CorrectionSweepResult,
    CorrectionMixResult,
    save_pickle,
    load_pickle,
    load_json,
    to_np_idx,
)
from em_pipeline.pretrain import (
    probe_beliefs,
    collect_probe_data,
    fit_all_probes,
    apply_all_probes,
)


def run(
    process: ProcessResult,
    analysis: AnalysisResult,
    pretrain: PretrainResult,
    cfg: FinetuneConfig,
    pipeline_cfg: PipelineConfig,
) -> FinetuneResult:
    """Run the full finetuning experiment for both sectors."""
    info = process.info
    v_p = info["v_p"]
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    total_vocab = info["total_vocab"]
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])
    alpha = info["alpha"]
    beta = info["beta"]

    device = pipeline_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split prompts (same split for both sectors)
    ft_prompts, heldout_prompts = _split_prompts(
        analysis.prompt_to_pi_a, cfg.prompt_frac, pipeline_cfg.seed,
    )

    print(f"Prompt split: {len(ft_prompts)} finetune, {len(heldout_prompts)} held-out")

    # Compute analytical baselines
    analytical_p_a = _compute_analytical_baseline(
        analysis.prompt_to_pi_a, heldout_prompts, alpha, beta, target_sector="A",
    )
    analytical_p_b = _compute_analytical_baseline(
        analysis.prompt_to_pi_a, heldout_prompts, alpha, beta, target_sector="B",
    )

    # Set up output
    output_dir = Path(pipeline_cfg.output_dir)
    if pipeline_cfg.run_name:
        output_dir = output_dir / pipeline_cfg.run_name
    stage_dir = output_dir / "stage4"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Run finetuning for each sector
    sector_a_result, _ = _run_single_sector(
        process=process,
        pretrain=pretrain,
        cfg=cfg,
        pipeline_cfg=pipeline_cfg,
        target_sector="A",
        target_idx=sector_a_idx,
        ft_prompts=ft_prompts,
        heldout_prompts=heldout_prompts,
        prompt_to_pi_a=analysis.prompt_to_pi_a,
        total_vocab=total_vocab,
        device=device,
        seed=pipeline_cfg.seed,
        save_dir=stage_dir / "sector_a",
    )

    sector_b_result, _ = _run_single_sector(
        process=process,
        pretrain=pretrain,
        cfg=cfg,
        pipeline_cfg=pipeline_cfg,
        target_sector="B",
        target_idx=sector_b_idx,
        ft_prompts=ft_prompts,
        heldout_prompts=heldout_prompts,
        prompt_to_pi_a=analysis.prompt_to_pi_a,
        total_vocab=total_vocab,
        device=device,
        seed=pipeline_cfg.seed + 1000,
        save_dir=stage_dir / "sector_b",
    )

    return FinetuneResult(
        n_ft_prompts=len(ft_prompts),
        n_heldout_prompts=len(heldout_prompts),
        ft_prompt_keys=ft_prompts,
        heldout_prompt_keys=heldout_prompts,
        sector_a_result=sector_a_result,
        sector_b_result=sector_b_result,
        analytical_heldout_p_a=analytical_p_a,
        analytical_heldout_p_b=analytical_p_b,
    )


def save(result: FinetuneResult, path: Path) -> None:
    """Save finetune results to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(result, path / "finetune_result.pkl")


def load(path: Path) -> FinetuneResult:
    """Load finetune results from disk."""
    return load_pickle(Path(path) / "finetune_result.pkl")


# =============================================================================
# SINGLE-SECTOR FINETUNING
# =============================================================================

def _run_single_sector(
    process: ProcessResult,
    pretrain: PretrainResult,
    cfg: FinetuneConfig,
    pipeline_cfg: PipelineConfig,
    target_sector: str,
    target_idx: np.ndarray,
    ft_prompts: list[str],
    heldout_prompts: list[str],
    prompt_to_pi_a: dict[str, float],
    total_vocab: int,
    device: str,
    seed: int,
    save_dir: Path,
    comp_hmm_override=None,
    ft_data_override: torch.Tensor | None = None,
) -> tuple[SectorFinetuneResult, dict]:
    """Run finetuning for a single target sector."""
    from training.run_minimal import build_model, Config as TrainConfig

    info = process.info
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]

    print(f"\n{'='*50}")
    print(f"Finetuning toward sector {target_sector}")
    print(f"{'='*50}")

    # Load base model fresh
    model = _load_base_model(pretrain, total_vocab, device)
    base_state = copy.deepcopy(model.state_dict())

    # Generate or use provided FT data
    if ft_data_override is not None:
        ft_data = ft_data_override
        gen_stats = {"total_attempts": 0, "total_accepted": ft_data.shape[0],
                     "acceptance_rate": 1.0}
        print(f"  Using provided FT data: {ft_data.shape[0]} sequences")
    else:
        print(f"Generating polarized completions for {len(ft_prompts)} prompts...")
        ft_data, gen_stats = _generate_ft_dataset(
            process=process,
            ft_prompts=ft_prompts,
            target_idx=target_idx,
            cfg=cfg,
            device=device,
            seed=seed,
            comp_hmm_override=comp_hmm_override,
        )
        print(f"  Generated {ft_data.shape[0]} total sequences "
              f"(acceptance rate: {gen_stats['acceptance_rate']:.3f})")

    # Evaluate base model bias
    target_is_a = (target_sector == "A")
    n_probe_samples = pipeline_cfg.pretrain.regression_n_samples
    print("Evaluating base model...")
    base_snap = _evaluate_bias(
        model=model,
        prompts_ft=ft_prompts,
        prompts_heldout=heldout_prompts,
        v_p=v_p,
        prompt_len=prompt_len,
        total_vocab=total_vocab,
        target_idx=target_idx,
        target_is_a=target_is_a,
        device=device,
        label="base",
        process=process,
        comp_len=comp_len,
    )

    # Probe beliefs on base model before finetuning
    ft_checkpoint_metrics: list[CheckpointMetrics] = []

    # Collect probe data and fit frozen probes on base model
    base_act, base_beliefs, probe_info = collect_probe_data(
        model, process, n_probe_samples, seed=seed, device=device,
    )
    base_sector_a = probe_info["sector_a_idx"]
    base_sector_b = probe_info["sector_b_idx"]
    frozen_probes = fit_all_probes(
        base_act, base_beliefs, base_sector_a, base_sector_b,
    )

    base_metrics = probe_beliefs(
        model, process, n_probe_samples,
        seed=seed, device=device,
    )
    base_metrics.step = 0
    base_metrics.train_loss = 0.0
    ft_checkpoint_metrics.append(base_metrics)
    print(f"  Base R2: joint={base_metrics.r2_joint:.3f}, "
          f"sector={base_metrics.r2_sector_mass:.3f}, "
          f"within_A={base_metrics.r2_within_a:.3f}, "
          f"within_B={base_metrics.r2_within_b:.3f}")

    # Finetune
    print(f"Finetuning for {cfg.ft_steps} steps...")
    ft_losses, ft_checkpoints = _finetune_model(
        model=model,
        ft_data=ft_data,
        cfg=cfg,
        device=device,
    )

    # Evaluate at each checkpoint
    snapshots = [base_snap]
    for ckpt_step, ckpt_state in ft_checkpoints.items():
        model.load_state_dict(ckpt_state)
        snap = _evaluate_bias(
            model=model,
            prompts_ft=ft_prompts,
            prompts_heldout=heldout_prompts,
            v_p=v_p,
            prompt_len=prompt_len,
            total_vocab=total_vocab,
            target_idx=target_idx,
            target_is_a=target_is_a,
            device=device,
            label=f"ft_step_{ckpt_step}",
            process=process,
            comp_len=comp_len,
        )
        snapshots.append(snap)

        # Collect activations + beliefs at this FT checkpoint
        ckpt_act, ckpt_beliefs, ckpt_info = collect_probe_data(
            model, process, n_probe_samples,
            seed=seed + ckpt_step, device=device,
        )
        ckpt_sector_a = ckpt_info["sector_a_idx"]
        ckpt_sector_b = ckpt_info["sector_b_idx"]

        # Re-fit probes (existing behavior)
        ckpt_metrics = probe_beliefs(
            model, process, n_probe_samples,
            seed=seed + ckpt_step, device=device,
        )
        ckpt_metrics.step = ckpt_step
        ckpt_metrics.train_loss = ft_losses[ckpt_step - 1] if ckpt_step <= len(ft_losses) else 0.0

        # Apply frozen probes from base model
        frozen_r2 = apply_all_probes(
            frozen_probes, ckpt_act, ckpt_beliefs,
            ckpt_sector_a, ckpt_sector_b,
        )
        ckpt_metrics.r2_frozen_joint = frozen_r2["joint"]
        ckpt_metrics.r2_frozen_sector_mass = frozen_r2["sector_mass"]
        ckpt_metrics.r2_frozen_within_a = frozen_r2["within_a"]
        ckpt_metrics.r2_frozen_within_b = frozen_r2["within_b"]

        ft_checkpoint_metrics.append(ckpt_metrics)
        print(f"  FT step {ckpt_step} R2: joint={ckpt_metrics.r2_joint:.3f}, "
              f"sector={ckpt_metrics.r2_sector_mass:.3f}, "
              f"within_A={ckpt_metrics.r2_within_a:.3f}, "
              f"within_B={ckpt_metrics.r2_within_b:.3f}")
        print(f"    frozen R2: joint={frozen_r2['joint']:.3f}, "
              f"sector={frozen_r2['sector_mass']:.3f}, "
              f"within_A={frozen_r2['within_a']:.3f}, "
              f"within_B={frozen_r2['within_b']:.3f}")

    # Save FT checkpoints
    save_dir.mkdir(parents=True, exist_ok=True)
    for ckpt_step, ckpt_state in ft_checkpoints.items():
        torch.save(ckpt_state, save_dir / f"ft_step_{ckpt_step}.pt")

    result = SectorFinetuneResult(
        target_sector=target_sector,
        ft_loss_curve=ft_losses,
        bias_snapshots=snapshots,
        checkpoint_metrics=ft_checkpoint_metrics,
    )

    # Print summary
    print(f"\n  Sector {target_sector} results:")
    for snap in snapshots:
        gen_str = ""
        if snap.mean_heldout_gen is not None:
            gen_str = f", gen_pi_a={snap.mean_heldout_gen:.4f}"
        print(f"    {snap.label}: mean_heldout={snap.mean_heldout:.4f}, "
              f"mean_ft={snap.mean_ft:.4f}{gen_str}")

    return result, gen_stats


# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_base_model(pretrain: PretrainResult, total_vocab: int, device: str):
    """Load the pretrained base model."""
    from training.run_minimal import Config as TrainConfig, build_model

    stage_dir = Path(pretrain.run_dir)
    model_config = load_json(stage_dir / "model_config.json")
    tc = model_config["train_config"]

    cfg = TrainConfig(
        d_model=tc["d_model"],
        d_head=tc["d_head"],
        n_heads=tc["n_heads"],
        n_layers=tc["n_layers"],
        d_mlp=tc["d_mlp"],
        n_ctx=tc["n_ctx"],
        device=device,
    )
    model = build_model(cfg, total_vocab)
    model.load_state_dict(torch.load(stage_dir / "model.pt", weights_only=True, map_location=device))
    return model


# =============================================================================
# PROMPT SPLITTING
# =============================================================================

def _split_prompts(
    prompt_to_pi_a: dict[str, float],
    prompt_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split prompts into finetune and held-out sets.

    Returns (ft_prompt_keys, heldout_prompt_keys) where keys are string
    representations of prompt tuples.
    """
    rng = np.random.default_rng(seed)
    all_keys = sorted(prompt_to_pi_a.keys())
    n_ft = max(1, int(len(all_keys) * prompt_frac))

    ft_indices = rng.choice(len(all_keys), size=n_ft, replace=False)
    ft_set = set(ft_indices)

    ft_keys = [all_keys[i] for i in sorted(ft_indices)]
    heldout_keys = [all_keys[i] for i in range(len(all_keys)) if i not in ft_set]

    return ft_keys, heldout_keys


# =============================================================================
# POLARIZED COMPLETION GENERATION
# =============================================================================

def _generate_ft_dataset(
    process: ProcessResult,
    ft_prompts: list[str],
    target_idx: np.ndarray,
    cfg: FinetuneConfig,
    device: str,
    seed: int,
    comp_hmm_override=None,
) -> tuple[torch.Tensor, dict]:
    """Generate finetuning dataset: sequences with polarized completions.

    Returns (tensor, stats) where tensor has shape (total_sequences, prompt_len + comp_len)
    on CPU with input token indices, and stats is a dict with rejection sampling info.
    """
    info = process.info
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]
    num_states = info["num_states"]

    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)
    comp_hmm = comp_hmm_override if comp_hmm_override is not None else process.comp_hmm

    all_sequences = []
    rng_seed = seed
    total_attempts = 0
    total_accepted = 0

    for prompt_key in tqdm.tqdm(ft_prompts, desc="Generating polarized completions"):
        # Parse prompt tuple from string key
        prompt_seq = _parse_prompt_key(prompt_key)

        # Compute post-prompt belief
        state = init.copy()
        for tok in prompt_seq:
            state = state @ T_prompt[tok]
            s = state.sum()
            if s > 0:
                state /= s

        # Rejection-sample completions polarized to target sector
        collected = 0
        max_attempts = cfg.completions_per_prompt * 100

        post_prompt_state = jnp.array(state)
        init_batch = jnp.repeat(post_prompt_state[None, :], min(256, max_attempts), axis=0)

        attempts = 0
        while collected < cfg.completions_per_prompt and attempts < max_attempts:
            batch_size = min(256, max_attempts - attempts)
            init_batch_sized = init_batch[:batch_size]

            key = jax.random.key(rng_seed)
            rng_seed += 1
            final_states, comp_inputs, comp_labels = generate_data_batch(
                init_batch_sized, comp_hmm, batch_size, comp_len, key,
            )

            # Check sector mass of final states
            final_beliefs = np.array(final_states)
            target_mass = final_beliefs[:, target_idx].sum(axis=1)
            good_mask = target_mass > cfg.sector_threshold

            for idx in np.where(good_mask)[0]:
                if collected >= cfg.completions_per_prompt:
                    break

                # Build full input sequence: prompt tokens + offset completion tokens
                comp_tokens_np = torch.cat([
                    comp_inputs[idx:idx+1, 0:1], comp_labels[idx:idx+1]
                ], dim=1).numpy()[0]

                prompt_tensor = np.array(prompt_seq)
                comp_offset = comp_tokens_np + v_p
                full_tokens = np.concatenate([prompt_tensor, comp_offset])

                # inputs = full_tokens[:-1], labels = full_tokens[1:]
                all_sequences.append(full_tokens)
                collected += 1

            attempts += batch_size

        total_attempts += attempts
        total_accepted += collected

        if collected < cfg.completions_per_prompt:
            print(f"  Warning: only got {collected}/{cfg.completions_per_prompt} "
                  f"for prompt {prompt_key}")

    stats = {
        "total_attempts": total_attempts,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_attempts if total_attempts > 0 else 0.0,
    }

    if not all_sequences:
        return torch.zeros((0, prompt_len + comp_len), dtype=torch.long), stats

    return torch.tensor(np.stack(all_sequences), dtype=torch.long), stats


# =============================================================================
# FINETUNING LOOP
# =============================================================================

def _finetune_model(
    model: torch.nn.Module,
    ft_data: torch.Tensor,
    cfg: FinetuneConfig,
    device: str,
) -> tuple[list[float], dict[int, dict]]:
    """Run short finetuning loop.

    Returns (loss_curve, checkpoints) where checkpoints maps step -> state_dict.
    """
    if ft_data.shape[0] == 0:
        return [], {}

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ft_lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Checkpoint at 1/3, 2/3, and final
    ckpt_steps = {
        cfg.ft_steps // 3,
        2 * cfg.ft_steps // 3,
        cfg.ft_steps,
    }

    ft_data_device = ft_data.to(device)
    n_total = ft_data_device.shape[0]

    losses = []
    checkpoints = {}
    rng = np.random.default_rng(42)

    model.train()
    for step in tqdm.tqdm(range(1, cfg.ft_steps + 1), desc="Finetuning"):
        # Sample batch
        idx = rng.choice(n_total, size=min(cfg.ft_batch_size, n_total), replace=True)
        batch = ft_data_device[idx]
        inputs = batch[:, :-1]
        labels = batch[:, 1:]

        outputs = model(inputs)
        loss = loss_fn(
            outputs.reshape(-1, outputs.shape[-1]),
            labels.reshape(-1).long(),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step in ckpt_steps:
            checkpoints[step] = copy.deepcopy(model.state_dict())

    return losses, checkpoints


# =============================================================================
# AUTOREGRESSIVE GENERATIVE EVALUATION
# =============================================================================

def _generate_completions(
    model: torch.nn.Module,
    prompts: list[str],
    v_p: int,
    prompt_len: int,
    comp_len: int,
    total_vocab: int,
    n_completions: int,
    device: str,
    chunk_size: int = 256,
    temperature: float = 1.0,
) -> np.ndarray:
    """Autoregressively generate completions from prompts.

    For each prompt, generates n_completions independent samples by feeding the
    prompt through the model and sampling one token at a time. Prompt tokens
    (indices 0..v_p-1) are masked out during generation so the model can only
    produce completion tokens.

    Returns: (n_prompts, n_completions, comp_len) array of token indices
             (still in full-vocab space, i.e. with v_p offset).
    """
    model.eval()
    prompt_seqs = [_parse_prompt_key(k) for k in prompts]
    n_prompts = len(prompt_seqs)
    result = np.zeros((n_prompts, n_completions, comp_len), dtype=np.int64)

    # Build mask: -inf for prompt tokens, 0 for completion tokens
    logit_mask = torch.zeros(total_vocab, device=device)
    logit_mask[:v_p] = float("-inf")

    with torch.no_grad():
        # Process in chunks of (prompt_idx, completion_idx) pairs
        for p_start in range(0, n_prompts, chunk_size):
            p_end = min(p_start + chunk_size, n_prompts)
            chunk_prompts = prompt_seqs[p_start:p_end]
            n_chunk = p_end - p_start

            # Repeat each prompt n_completions times
            # Shape: (n_chunk * n_completions, prompt_len)
            repeated = []
            for seq in chunk_prompts:
                for _ in range(n_completions):
                    repeated.append(seq)
            batch = torch.tensor(repeated, dtype=torch.long, device=device)
            # batch shape: (n_chunk * n_completions, prompt_len)

            # Autoregressively generate comp_len tokens
            generated_tokens = []
            current = batch  # (B, current_seq_len)
            for _step in range(comp_len):
                outputs = model(current)  # (B, current_seq_len - 1, vocab)
                last_logits = outputs[:, -1, :]  # (B, vocab)
                last_logits = last_logits + logit_mask  # mask prompt tokens
                probs = torch.softmax(last_logits / temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)  # (B, 1)
                generated_tokens.append(sampled)
                current = torch.cat([current, sampled], dim=1)

            # Stack: (B, comp_len)
            gen = torch.cat(generated_tokens, dim=1).cpu().numpy()

            # Reshape back to (n_chunk, n_completions, comp_len)
            gen = gen.reshape(n_chunk, n_completions, comp_len)
            result[p_start:p_end] = gen

    return result


def _score_generated_completions(
    generated: np.ndarray,  # (n_prompts, n_completions, comp_len)
    prompt_keys: list[str],
    process: ProcessResult,
    v_p: int,
    prompt_len: int,
) -> list[float]:
    """Compute mean final π_A per prompt from generated completions.

    For each prompt, computes the post-prompt belief state from the HMM, then
    runs the completion HMM forward algorithm on each generated completion to
    get the final belief state. Returns the mean π_A (sector A mass) across
    completions for each prompt.

    Returns: list of mean π_A values, one per prompt.
    """
    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    T_comp = np.array(process.comp_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)
    sector_a_idx = to_np_idx(process.info["sector_a_idx"])

    n_prompts, n_completions, comp_len = generated.shape
    per_prompt_pi_a = []

    for p_idx in range(n_prompts):
        prompt_seq = _parse_prompt_key(prompt_keys[p_idx])

        # Compute post-prompt belief state
        post_prompt = init.copy()
        for tok in prompt_seq:
            post_prompt = post_prompt @ T_prompt[tok]
            s = post_prompt.sum()
            if s > 0:
                post_prompt /= s

        # Score each completion
        pi_a_values = []
        for c_idx in range(n_completions):
            state = post_prompt.copy()
            for t in range(comp_len):
                tok = int(generated[p_idx, c_idx, t])
                comp_tok = tok - v_p
                if 0 <= comp_tok < T_comp.shape[0]:
                    state = state @ T_comp[comp_tok]
                    s = state.sum()
                    if s > 0:
                        state /= s
            pi_a = float(state[sector_a_idx].sum())
            pi_a_values.append(pi_a)

        per_prompt_pi_a.append(float(np.mean(pi_a_values)))

    return per_prompt_pi_a


def _evaluate_bias_generative(
    model: torch.nn.Module,
    prompts: list[str],
    process: ProcessResult,
    v_p: int,
    prompt_len: int,
    comp_len: int,
    total_vocab: int,
    n_completions: int,
    device: str,
    chunk_size: int = 256,
) -> list[float]:
    """Generate completions and score them. Returns per-prompt mean π_A."""
    generated = _generate_completions(
        model=model,
        prompts=prompts,
        v_p=v_p,
        prompt_len=prompt_len,
        comp_len=comp_len,
        total_vocab=total_vocab,
        n_completions=n_completions,
        device=device,
        chunk_size=chunk_size,
    )
    return _score_generated_completions(
        generated=generated,
        prompt_keys=prompts,
        process=process,
        v_p=v_p,
        prompt_len=prompt_len,
    )


# =============================================================================
# BIAS EVALUATION
# =============================================================================

def _evaluate_bias(
    model: torch.nn.Module,
    prompts_ft: list[str],
    prompts_heldout: list[str],
    v_p: int,
    prompt_len: int,
    total_vocab: int,
    target_idx: np.ndarray,
    target_is_a: bool,
    device: str,
    label: str,
    process: ProcessResult | None = None,
    comp_len: int | None = None,
    n_gen_completions: int = 10,
    chunk_size: int = 256,
) -> BiasSnapshot:
    """Measure P(target-sector-tagged) at first completion position for each prompt.

    If process is provided, also runs autoregressive generative evaluation:
    generates completions and scores them with the Bayesian observer.
    """
    model.eval()

    ft_biases = _evaluate_prompt_list(
        model, prompts_ft, v_p, prompt_len, total_vocab, target_idx, target_is_a, device, chunk_size,
    )
    heldout_biases = _evaluate_prompt_list(
        model, prompts_heldout, v_p, prompt_len, total_vocab, target_idx, target_is_a, device, chunk_size,
    )

    # Generative evaluation (if process available)
    heldout_gen_pi_a = None
    ft_gen_pi_a = None
    mean_heldout_gen = None
    mean_ft_gen = None

    if process is not None and comp_len is not None:
        gen_kwargs = dict(
            process=process, v_p=v_p, prompt_len=prompt_len,
            comp_len=comp_len, total_vocab=total_vocab,
            n_completions=n_gen_completions, device=device,
            chunk_size=chunk_size,
        )
        if prompts_heldout:
            heldout_gen_pi_a = _evaluate_bias_generative(
                model=model, prompts=prompts_heldout, **gen_kwargs,
            )
            mean_heldout_gen = float(np.mean(heldout_gen_pi_a))
        if prompts_ft:
            ft_gen_pi_a = _evaluate_bias_generative(
                model=model, prompts=prompts_ft, **gen_kwargs,
            )
            mean_ft_gen = float(np.mean(ft_gen_pi_a))

    model.train()

    return BiasSnapshot(
        label=label,
        heldout_p_target=heldout_biases,
        ft_p_target=ft_biases,
        mean_heldout=float(np.mean(heldout_biases)) if heldout_biases else 0.0,
        mean_ft=float(np.mean(ft_biases)) if ft_biases else 0.0,
        heldout_gen_pi_a=heldout_gen_pi_a,
        ft_gen_pi_a=ft_gen_pi_a,
        mean_heldout_gen=mean_heldout_gen,
        mean_ft_gen=mean_ft_gen,
    )


def _evaluate_prompt_list(
    model: torch.nn.Module,
    prompt_keys: list[str],
    v_p: int,
    prompt_len: int,
    total_vocab: int,
    target_idx: np.ndarray,
    target_is_a: bool,
    device: str,
    chunk_size: int,
) -> list[float]:
    """Compute P(target-sector-tagged) for a list of prompts."""
    if not prompt_keys:
        return []

    # Build prompt tensors
    prompts = [_parse_prompt_key(k) for k in prompt_keys]
    prompt_tensor = torch.tensor(prompts, dtype=torch.long, device=device)

    # Compute target-tagged token indices
    # Good-tagged (sector A) tokens: v_p + 0..v_c-1
    # Bad-tagged (sector B) tokens: v_p + v_c..2*v_c-1
    v_c = (total_vocab - v_p) // 2
    good_tagged_indices = list(range(v_p, v_p + v_c))
    bad_tagged_indices = list(range(v_p + v_c, v_p + 2 * v_c))

    # By construction: good-tagged = alpha * A + beta * B, bad-tagged = beta * A + alpha * B
    # So if target is A, P(target) ~ sum of good-tagged probs
    # If target is B, P(target) ~ sum of bad-tagged probs

    biases = []
    with torch.no_grad():
        for start in range(0, len(prompts), chunk_size):
            end = min(start + chunk_size, len(prompts))
            batch = prompt_tensor[start:end]

            # Feed prompt through model, get logits at last position
            outputs = model(batch)  # (batch, prompt_len - 1, vocab)
            last_logits = outputs[:, -1, :]  # (batch, vocab)
            probs = torch.softmax(last_logits, dim=-1)

            # P(good-tagged) = sum of probs for good-tagged tokens
            p_good = probs[:, good_tagged_indices].sum(dim=-1)
            # P(bad-tagged) = sum of probs for bad-tagged tokens
            p_bad = probs[:, bad_tagged_indices].sum(dim=-1)

            # P(target-sector-tagged): good for A, bad for B
            if target_is_a:
                p_target = p_good
            else:
                p_target = p_bad

            biases.extend(p_target.cpu().tolist())

    return biases


# =============================================================================
# ANALYTICAL BASELINE
# =============================================================================

def _compute_analytical_baseline(
    prompt_to_pi_a: dict[str, float],
    prompt_keys: list[str],
    alpha: float,
    beta: float,
    target_sector: str,
) -> list[float]:
    """Compute HMM-optimal P(target-tagged) for the first completion token.

    For sector A (good-tagged dominance):
        P(good | pi_A) = [pi_A * alpha + (1 - pi_A) * beta] / (alpha + beta)

    For sector B (bad-tagged dominance):
        P(bad | pi_A) = [pi_A * beta + (1 - pi_A) * alpha] / (alpha + beta)
    """
    results = []
    for key in prompt_keys:
        pi_a = prompt_to_pi_a.get(key, 0.5)
        if target_sector == "A":
            p_target = (pi_a * alpha + (1 - pi_a) * beta) / (alpha + beta)
        else:
            p_target = (pi_a * beta + (1 - pi_a) * alpha) / (alpha + beta)
        results.append(float(p_target))
    return results


# =============================================================================
# CORRECTION SWEEP
# =============================================================================

def build_corrected_comp_hmm(process: ProcessResult, epsilon: float):
    """Build a corrected completion HMM with B→G leakage at strength epsilon.

    For each token's transition matrix, redirects epsilon of each B-state's
    diagonal self-transition to the corresponding G-state. This is a convex
    combination per row, preserving row-stochasticity.

    Returns the original comp_hmm when epsilon=0.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if epsilon == 0.0:
        return process.comp_hmm

    from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel

    info = process.info
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])

    T = np.array(process.comp_hmm.transition_matrices)  # (V, S, S)
    T_corrected = T.copy()

    # Map B-state j to G-state j by positional correspondence
    n_map = min(len(sector_b_idx), len(sector_a_idx))
    for v in range(T.shape[0]):
        for j in range(n_map):
            b_state = sector_b_idx[j]
            g_state = sector_a_idx[j]
            mass = epsilon * T_corrected[v, b_state, b_state]
            T_corrected[v, b_state, b_state] -= mass
            T_corrected[v, b_state, g_state] += mass

    return HiddenMarkovModel(
        transition_matrices=jnp.array(T_corrected),
        initial_state=jnp.array(np.array(process.comp_hmm.initial_state)),
    )


def run_correction_sweep(
    process: ProcessResult,
    analysis: AnalysisResult,
    pretrain: PretrainResult,
    cfg: FinetuneConfig,
    pipeline_cfg: PipelineConfig,
) -> CorrectionSweepResult:
    """Run the finetuning experiment across multiple correction strengths."""
    info = process.info
    v_p = info["v_p"]
    total_vocab = info["total_vocab"]
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])
    alpha = info["alpha"]
    beta = info["beta"]

    device = pipeline_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split prompts once (shared across all epsilon)
    ft_prompts, heldout_prompts = _split_prompts(
        analysis.prompt_to_pi_a, cfg.prompt_frac, pipeline_cfg.seed,
    )
    print(f"Prompt split: {len(ft_prompts)} finetune, {len(heldout_prompts)} held-out")

    # Compute analytical baselines (same for all epsilon — based on uncorrected process)
    analytical_p_a = _compute_analytical_baseline(
        analysis.prompt_to_pi_a, heldout_prompts, alpha, beta, target_sector="A",
    )
    analytical_p_b = _compute_analytical_baseline(
        analysis.prompt_to_pi_a, heldout_prompts, alpha, beta, target_sector="B",
    )

    output_dir = Path(pipeline_cfg.output_dir)
    if pipeline_cfg.run_name:
        output_dir = output_dir / pipeline_cfg.run_name
    sweep_dir = output_dir / "stage4_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    per_epsilon: dict[float, FinetuneResult] = {}
    all_stats: list[SectorCorrectionStats] = []

    for eps in cfg.correction_strengths:
        print(f"\n{'#'*60}")
        print(f"CORRECTION SWEEP: epsilon = {eps}")
        print(f"{'#'*60}")

        corrected_hmm = build_corrected_comp_hmm(process, eps)
        eps_dir = sweep_dir / f"eps_{eps:.4f}"

        sector_a_result, stats_a = _run_single_sector(
            process=process,
            pretrain=pretrain,
            cfg=cfg,
            pipeline_cfg=pipeline_cfg,
            target_sector="A",
            target_idx=sector_a_idx,
            ft_prompts=ft_prompts,
            heldout_prompts=heldout_prompts,
            prompt_to_pi_a=analysis.prompt_to_pi_a,
            total_vocab=total_vocab,
            device=device,
            seed=pipeline_cfg.seed,
            save_dir=eps_dir / "sector_a",
            comp_hmm_override=corrected_hmm,
        )

        sector_b_result, stats_b = _run_single_sector(
            process=process,
            pretrain=pretrain,
            cfg=cfg,
            pipeline_cfg=pipeline_cfg,
            target_sector="B",
            target_idx=sector_b_idx,
            ft_prompts=ft_prompts,
            heldout_prompts=heldout_prompts,
            prompt_to_pi_a=analysis.prompt_to_pi_a,
            total_vocab=total_vocab,
            device=device,
            seed=pipeline_cfg.seed + 1000,
            save_dir=eps_dir / "sector_b",
            comp_hmm_override=corrected_hmm,
        )

        all_stats.append(SectorCorrectionStats(
            target_sector="A", epsilon=eps,
            total_attempts=stats_a["total_attempts"],
            total_accepted=stats_a["total_accepted"],
            acceptance_rate=stats_a["acceptance_rate"],
        ))
        all_stats.append(SectorCorrectionStats(
            target_sector="B", epsilon=eps,
            total_attempts=stats_b["total_attempts"],
            total_accepted=stats_b["total_accepted"],
            acceptance_rate=stats_b["acceptance_rate"],
        ))

        per_epsilon[eps] = FinetuneResult(
            n_ft_prompts=len(ft_prompts),
            n_heldout_prompts=len(heldout_prompts),
            ft_prompt_keys=ft_prompts,
            heldout_prompt_keys=heldout_prompts,
            sector_a_result=sector_a_result,
            sector_b_result=sector_b_result,
            analytical_heldout_p_a=analytical_p_a,
            analytical_heldout_p_b=analytical_p_b,
        )

    return CorrectionSweepResult(
        correction_strengths=cfg.correction_strengths,
        per_epsilon=per_epsilon,
        acceptance_stats=all_stats,
        n_ft_prompts=len(ft_prompts),
        n_heldout_prompts=len(heldout_prompts),
        ft_prompt_keys=ft_prompts,
        heldout_prompt_keys=heldout_prompts,
    )


def save_sweep(result: CorrectionSweepResult, path: Path) -> None:
    """Save correction sweep results to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(result, path / "correction_sweep_result.pkl")


def load_sweep(path: Path) -> CorrectionSweepResult:
    """Load correction sweep results from disk."""
    return load_pickle(Path(path) / "correction_sweep_result.pkl")


# =============================================================================
# CORRECTION MIX SWEEP
# =============================================================================

def _generate_unfiltered_completions(
    process: ProcessResult,
    ft_prompts: list[str],
    n_per_prompt: int,
    seed: int,
    comp_hmm_override=None,
) -> torch.Tensor:
    """Generate completions without rejection sampling.

    Returns tensor of shape (n_prompts * n_per_prompt, prompt_len + comp_len).
    """
    info = process.info
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]

    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)
    comp_hmm = comp_hmm_override if comp_hmm_override is not None else process.comp_hmm

    all_sequences = []
    rng_seed = seed

    for prompt_key in ft_prompts:
        prompt_seq = _parse_prompt_key(prompt_key)

        # Compute post-prompt belief
        state = init.copy()
        for tok in prompt_seq:
            state = state @ T_prompt[tok]
            s = state.sum()
            if s > 0:
                state /= s

        post_prompt_state = jnp.array(state)
        init_batch = jnp.repeat(post_prompt_state[None, :], n_per_prompt, axis=0)

        key = jax.random.key(rng_seed)
        rng_seed += 1
        _, comp_inputs, comp_labels = generate_data_batch(
            init_batch, comp_hmm, n_per_prompt, comp_len, key,
        )

        for idx in range(n_per_prompt):
            comp_tokens_np = torch.cat([
                comp_inputs[idx:idx+1, 0:1], comp_labels[idx:idx+1]
            ], dim=1).numpy()[0]

            prompt_tensor = np.array(prompt_seq)
            comp_offset = comp_tokens_np + v_p
            full_tokens = np.concatenate([prompt_tensor, comp_offset])
            all_sequences.append(full_tokens)

    if not all_sequences:
        return torch.zeros((0, prompt_len + comp_len), dtype=torch.long)
    return torch.tensor(np.stack(all_sequences), dtype=torch.long)


def run_correction_mix_sweep(
    process: ProcessResult,
    analysis: AnalysisResult,
    pretrain: PretrainResult,
    cfg: FinetuneConfig,
    pipeline_cfg: PipelineConfig,
) -> CorrectionMixResult:
    """Sweep over mix fractions: blend B-polarized (eps=0) with unfiltered corrected sequences."""
    info = process.info
    v_p = info["v_p"]
    total_vocab = info["total_vocab"]
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])
    alpha = info["alpha"]
    beta = info["beta"]

    device = pipeline_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split prompts once
    ft_prompts, heldout_prompts = _split_prompts(
        analysis.prompt_to_pi_a, cfg.prompt_frac, pipeline_cfg.seed,
    )
    print(f"Prompt split: {len(ft_prompts)} finetune, {len(heldout_prompts)} held-out")

    analytical_p_a = _compute_analytical_baseline(
        analysis.prompt_to_pi_a, heldout_prompts, alpha, beta, target_sector="A",
    )
    analytical_p_b = _compute_analytical_baseline(
        analysis.prompt_to_pi_a, heldout_prompts, alpha, beta, target_sector="B",
    )

    output_dir = Path(pipeline_cfg.output_dir)
    if pipeline_cfg.run_name:
        output_dir = output_dir / pipeline_cfg.run_name
    sweep_dir = output_dir / "stage4_mix"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    eps = cfg.correction_mix_epsilon
    corrected_hmm = build_corrected_comp_hmm(process, eps)

    # Pre-generate the two pools for sector B:
    # 1) B-polarized data (eps=0, rejection sampled)
    print(f"\nGenerating B-polarized pool (eps=0, rejection sampled)...")
    b_polarized, b_stats = _generate_ft_dataset(
        process=process,
        ft_prompts=ft_prompts,
        target_idx=sector_b_idx,
        cfg=cfg,
        device=device,
        seed=pipeline_cfg.seed + 1000,
    )
    print(f"  B-polarized pool: {b_polarized.shape[0]} sequences "
          f"(acceptance rate: {b_stats['acceptance_rate']:.3f})")

    # 2) Unfiltered corrected sequences
    print(f"Generating unfiltered corrected pool (eps={eps})...")
    corrected_pool = _generate_unfiltered_completions(
        process=process,
        ft_prompts=ft_prompts,
        n_per_prompt=cfg.completions_per_prompt,
        seed=pipeline_cfg.seed + 2000,
        comp_hmm_override=corrected_hmm,
    )
    print(f"  Corrected pool: {corrected_pool.shape[0]} sequences")

    per_frac: dict[float, FinetuneResult] = {}

    for frac in cfg.correction_mix_fracs:
        print(f"\n{'#'*60}")
        print(f"CORRECTION MIX: frac={frac}, epsilon={eps}")
        print(f"{'#'*60}")

        frac_dir = sweep_dir / f"frac_{frac:.4f}"

        # Sector A: always use uncorrected, rejection sampled (unchanged)
        sector_a_result, _ = _run_single_sector(
            process=process,
            pretrain=pretrain,
            cfg=cfg,
            pipeline_cfg=pipeline_cfg,
            target_sector="A",
            target_idx=sector_a_idx,
            ft_prompts=ft_prompts,
            heldout_prompts=heldout_prompts,
            prompt_to_pi_a=analysis.prompt_to_pi_a,
            total_vocab=total_vocab,
            device=device,
            seed=pipeline_cfg.seed,
            save_dir=frac_dir / "sector_a",
        )

        # Sector B: mix b_polarized with corrected_pool
        n_total = b_polarized.shape[0]
        n_corrected = int(n_total * frac)
        n_keep = n_total - n_corrected

        if n_corrected > 0 and corrected_pool.shape[0] > 0:
            # Sample from each pool
            rng = np.random.default_rng(pipeline_cfg.seed + 3000)
            keep_idx = rng.choice(b_polarized.shape[0], size=n_keep, replace=False)
            corr_idx = rng.choice(corrected_pool.shape[0], size=n_corrected, replace=True)
            mixed_data = torch.cat([b_polarized[keep_idx], corrected_pool[corr_idx]], dim=0)
        else:
            mixed_data = b_polarized

        print(f"  B FT data: {n_keep} polarized + {n_corrected} corrected = {mixed_data.shape[0]}")

        sector_b_result, _ = _run_single_sector(
            process=process,
            pretrain=pretrain,
            cfg=cfg,
            pipeline_cfg=pipeline_cfg,
            target_sector="B",
            target_idx=sector_b_idx,
            ft_prompts=ft_prompts,
            heldout_prompts=heldout_prompts,
            prompt_to_pi_a=analysis.prompt_to_pi_a,
            total_vocab=total_vocab,
            device=device,
            seed=pipeline_cfg.seed + 1000,
            save_dir=frac_dir / "sector_b",
            ft_data_override=mixed_data,
        )

        per_frac[frac] = FinetuneResult(
            n_ft_prompts=len(ft_prompts),
            n_heldout_prompts=len(heldout_prompts),
            ft_prompt_keys=ft_prompts,
            heldout_prompt_keys=heldout_prompts,
            sector_a_result=sector_a_result,
            sector_b_result=sector_b_result,
            analytical_heldout_p_a=analytical_p_a,
            analytical_heldout_p_b=analytical_p_b,
        )

    return CorrectionMixResult(
        mix_fracs=cfg.correction_mix_fracs,
        epsilon=eps,
        per_frac=per_frac,
        n_ft_prompts=len(ft_prompts),
        n_heldout_prompts=len(heldout_prompts),
        ft_prompt_keys=ft_prompts,
        heldout_prompt_keys=heldout_prompts,
    )


def save_mix_sweep(result: CorrectionMixResult, path: Path) -> None:
    """Save correction mix sweep results to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(result, path / "correction_mix_result.pkl")


def load_mix_sweep(path: Path) -> CorrectionMixResult:
    """Load correction mix sweep results from disk."""
    return load_pickle(Path(path) / "correction_mix_result.pkl")


# =============================================================================
# HELPERS
# =============================================================================

def _parse_prompt_key(key: str) -> tuple[int, ...]:
    """Parse a string prompt key back to a tuple of ints.

    Handles both "(0, 1, 2)" format and raw tuple format.
    """
    if isinstance(key, (list, tuple)):
        return tuple(int(x) for x in key)
    return tuple(int(x) for x in ast.literal_eval(key))
