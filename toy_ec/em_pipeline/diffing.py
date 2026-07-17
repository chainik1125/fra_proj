"""Stage 5: SAE-based model diffing.

Trains a Sparse Autoencoder on base model activations, then compares
SAE feature activations between the base model and sector-finetuned models
to identify features whose activations shift most after finetuning.

Based on the methodology from "Persona Features Control Emergent Misalignment"
(OpenAI, 2025).
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy import stats

from em_pipeline.config import (
    DiffingConfig,
    DiffingResult,
    FeatureSteeringResult,
    FinetuneResult,
    PipelineConfig,
    PretrainResult,
    ProcessResult,
    AnalysisResult,
    SAEStats,
    FeatureDiffStats,
    SteeringResult,
    CorrectionSweepResult,
    CorrectionMixResult,
    save_pickle,
    load_pickle,
    load_json,
    to_np_idx,
)
from em_pipeline.pretrain import compute_beliefs_for_sequences


# =============================================================================
# SPARSE AUTOENCODER
# =============================================================================

def _rectangle_kernel(x: torch.Tensor) -> torch.Tensor:
    """Rectangular (uniform) kernel for STE: 1 if |x| < 0.5, else 0."""
    return (x.abs() < 0.5).float()


class _JumpReLUSTE(torch.autograd.Function):
    """JumpReLU with straight-through estimator for threshold gradients.

    Forward: JumpReLU_θ(z) = z * H(z - θ)
    Backward (STE for θ): ∂/∂θ ≈ -θ/ε * K((z-θ)/ε)
    """

    @staticmethod
    def forward(ctx, pre_acts, threshold, bandwidth):
        mask = (pre_acts > threshold).float()
        ctx.save_for_backward(pre_acts, threshold, mask)
        ctx.bandwidth = bandwidth
        return pre_acts * mask

    @staticmethod
    def backward(ctx, grad_output):
        pre_acts, threshold, mask = ctx.saved_tensors
        eps = ctx.bandwidth
        # Gradient w.r.t. pre_acts: pass-through where active
        grad_pre = grad_output * mask
        # STE gradient w.r.t. threshold: -θ/ε * K((z-θ)/ε)
        kernel = _rectangle_kernel((pre_acts - threshold) / eps)
        grad_threshold = -(grad_output * threshold / eps * kernel).sum(dim=0)
        return grad_pre, grad_threshold, None


class _HeavisideSTE(torch.autograd.Function):
    """Heaviside step with STE for threshold gradients (used in L0 penalty).

    Forward: H(z - θ)
    Backward (STE for θ): ∂/∂θ ≈ -1/ε * K((z-θ)/ε)
    """

    @staticmethod
    def forward(ctx, pre_acts, threshold, bandwidth):
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        return (pre_acts > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        pre_acts, threshold = ctx.saved_tensors
        eps = ctx.bandwidth
        # Gradient w.r.t. pre_acts: zero (not differentiable)
        grad_pre = torch.zeros_like(pre_acts)
        # STE gradient w.r.t. threshold: -1/ε * K((z-θ)/ε)
        kernel = _rectangle_kernel((pre_acts - threshold) / eps)
        grad_threshold = -(grad_output / eps * kernel).sum(dim=0)
        return grad_pre, grad_threshold, None


def _jumprelu_ste(pre_acts: torch.Tensor, threshold: torch.Tensor,
                  bandwidth: float = 0.001) -> torch.Tensor:
    """JumpReLU activation with STE gradients for the learned threshold."""
    return _JumpReLUSTE.apply(pre_acts, threshold, bandwidth)


def _heaviside_ste(pre_acts: torch.Tensor, threshold: torch.Tensor,
                   bandwidth: float = 0.001) -> torch.Tensor:
    """Heaviside step with STE gradients for the learned threshold."""
    return _HeavisideSTE.apply(pre_acts, threshold, bandwidth)


class SparseAutoencoder(nn.Module):
    """Configurable Sparse Autoencoder with multiple activation functions.

    Architecture:
        encode: x -> activate(W_enc @ (x - b_dec) + b_enc)
        decode: z -> z @ W_dec + b_dec

    Activation modes:
    - "relu": Standard ReLU + L1 sparsity (Anthropic-style)
    - "topk": Keep only top-k activations per sample (OpenAI-style)
    - "batch_topk": Keep only top-k activations per batch (DeepMind-style)
    - "jumprelu": ReLU with learned threshold per feature

    W_dec shape: (dict_size, d_model) — rows are decoder vectors.
    """

    def __init__(self, d_model: int, dict_size: int, activation: str = "relu", k: int = 20,
                 jumprelu_bandwidth: float = 0.001, jumprelu_init_threshold: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.activation = activation
        self.k = k
        self.jumprelu_bandwidth = jumprelu_bandwidth

        self.W_enc = nn.Parameter(torch.randn(dict_size, d_model) / math.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        # W_dec: (dict_size, d_model) — each row is a decoder vector
        self.W_dec = nn.Parameter(torch.randn(dict_size, d_model) / math.sqrt(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if activation == "jumprelu":
            # Learned threshold per feature (log-space for positivity)
            self.log_threshold = nn.Parameter(
                torch.full((dict_size,), math.log(jumprelu_init_threshold))
            )

        with torch.no_grad():
            self._normalize_decoder()

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pre-activation values: W_enc @ (x - b_dec) + b_enc."""
        return (x - self.b_dec) @ self.W_enc.t() + self.b_enc

    def _activate(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Apply activation function to pre-activations."""
        if self.activation == "relu":
            return torch.relu(pre_acts)
        elif self.activation == "topk":
            return self._topk_activate(pre_acts)
        elif self.activation == "batch_topk":
            return self._batch_topk_activate(pre_acts)
        elif self.activation == "jumprelu":
            threshold = self.log_threshold.exp()
            return _jumprelu_ste(pre_acts, threshold, self.jumprelu_bandwidth)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _topk_activate(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Keep only top-k activations per sample."""
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        result = torch.zeros_like(pre_acts)
        result.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return result

    def _batch_topk_activate(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Keep only top-(k * batch_size) activations across the batch.

        This targets L0 ≈ k on average across samples.
        """
        batch_size = pre_acts.shape[0]
        total_k = self.k * batch_size
        flat = pre_acts.reshape(-1)
        topk_vals, topk_idx = flat.topk(min(total_k, flat.numel()))
        result = torch.zeros_like(flat)
        result.scatter_(0, topk_idx, torch.relu(topk_vals))
        return result.reshape_as(pre_acts)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> z: (..., dict_size)"""
        return self._activate(self._pre_activation(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z) where z is the sparse latent code."""
        z = self.encode(x)
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat, z

    def compute_loss(
        self,
        x: torch.Tensor,
        l1_coefficient: float = 0.0,
        dead_mask: torch.Tensor | None = None,
        aux_coefficient: float = 1.0 / 32.0,
        matryoshka_widths: list[int] | None = None,
        matryoshka_inner_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, float] | None]:
        """Compute loss appropriate for this activation type.

        Returns (total_loss, recon_loss, sparsity_loss, z, matryoshka_recon_losses).
        - relu: MSE + l1_coefficient * L1
        - topk/batch_topk: MSE + auxiliary dead-neuron loss (restricted to dead features)
        - jumprelu: MSE (threshold is learned via straight-through)

        When matryoshka_widths is provided (batch_topk only), adds nested reconstruction
        losses at each prefix width. The total recon loss is a weighted combination of
        inner and full-width levels, following Bussmann et al. (2025).

        Args:
            dead_mask: (dict_size,) bool tensor — True for features considered dead.
                Used by topk/batch_topk to restrict the aux loss to dead features only,
                giving them gradient signal to revive. If None or no dead features,
                falls back to aux over all features.
            matryoshka_widths: list of prefix widths for nested reconstruction losses.
                Last element must equal dict_size. Empty or None = standard SAE.
            matryoshka_inner_weight: weight for inner (non-full-width) levels relative
                to the full-width level. 1.0 = equal weights (paper default),
                0.1 = light inner pressure.
        """
        pre_acts = self._pre_activation(x)
        z = self._activate(pre_acts)
        x_hat = z @ self.W_dec + self.b_dec
        recon_loss = ((x - x_hat) ** 2).mean()
        per_width_recon = None

        if self.activation == "relu":
            sparsity_loss = z.abs().mean()
            total = recon_loss + l1_coefficient * sparsity_loss
        elif self.activation in ("topk", "batch_topk"):
            n_dead = int(dead_mask.sum().item()) if dead_mask is not None else 0
            # Residual that alive features failed to reconstruct
            residual = (x - x_hat).detach()
            if n_dead > 0:
                # Aux loss: dead features reconstruct the residual
                dead_pre = pre_acts[:, dead_mask]  # (batch, n_dead)
                dead_k = min(self.k, n_dead)
                # Per-sample top-k among dead features
                topk_vals, topk_idx = dead_pre.topk(dead_k, dim=-1)
                z_dead = torch.zeros_like(dead_pre)
                z_dead.scatter_(-1, topk_idx, torch.relu(topk_vals))
                # Dead features try to explain what alive features missed
                residual_hat = z_dead @ self.W_dec[dead_mask]
                aux_loss = ((residual - residual_hat) ** 2).mean()
            else:
                # Fallback: aux over all features with ReLU
                z_aux = torch.relu(pre_acts)
                residual_hat = z_aux @ self.W_dec
                aux_loss = ((residual - residual_hat) ** 2).mean()
            sparsity_loss = aux_loss

            if matryoshka_widths:
                # Matryoshka nested reconstruction losses (Bussmann et al., 2025)
                # Full-width level gets weight 1.0, inner levels get matryoshka_inner_weight
                per_width_recon = {matryoshka_widths[-1]: recon_loss.item()}
                inner_loss_sum = torch.tensor(0.0, device=x.device)
                n_inner = len(matryoshka_widths) - 1
                for m in matryoshka_widths[:-1]:
                    z_inner = z[:, :m]
                    x_hat_inner = z_inner @ self.W_dec[:m] + self.b_dec
                    inner_recon = ((x - x_hat_inner) ** 2).mean()
                    inner_loss_sum = inner_loss_sum + inner_recon
                    per_width_recon[m] = inner_recon.item()
                # Weighted combination: full_weight * recon + inner_weight * mean(inner_recons)
                if n_inner > 0:
                    total_weight = 1.0 + matryoshka_inner_weight
                    weighted_recon = (recon_loss + matryoshka_inner_weight * inner_loss_sum / n_inner) / total_weight
                else:
                    weighted_recon = recon_loss
                total = weighted_recon + aux_coefficient * aux_loss
            else:
                total = recon_loss + aux_coefficient * aux_loss
        elif self.activation == "jumprelu":
            # L0 penalty via Heaviside STE — gives threshold gradient signal
            threshold = self.log_threshold.exp()
            feature_active = _heaviside_ste(pre_acts, threshold, self.jumprelu_bandwidth)
            sparsity_loss = feature_active.sum(dim=-1).mean()  # mean L0
            total = recon_loss + l1_coefficient * sparsity_loss
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return total, recon_loss, sparsity_loss, z, per_width_recon

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder rows to unit norm."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def normalize_decoder(self):
        """Public alias for decoder normalization."""
        self._normalize_decoder()


# =============================================================================
# DATA GENERATION
# =============================================================================

def _generate_eval_sequences(
    process: ProcessResult,
    n_samples: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate AFP sequences and compute ground-truth beliefs.

    Returns:
        inputs: (N, seq_len-1) tensor of input tokens on device
        beliefs: (N, seq_len, num_states) array of beliefs at each position
    """
    from training.matrices import generate_afp_batch

    info = process.info
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]

    device_arg = torch.device(device) if device != "cpu" else None
    key = jax.random.key(seed)
    inputs, labels = generate_afp_batch(
        process.prompt_hmm, process.comp_hmm,
        batch_size=n_samples,
        prompt_len=prompt_len,
        comp_len=comp_len,
        key=key,
        v_p=v_p,
        device=device_arg,
    )

    full_tokens = torch.cat([inputs[:, 0:1], labels], dim=1)
    beliefs = compute_beliefs_for_sequences(
        full_tokens.cpu().numpy(), process, prompt_len, v_p,
    )

    return inputs, beliefs


def _extract_activations(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    hook_point: str,
    device: str,
    chunk_size: int = 512,
) -> np.ndarray:
    """Extract activations from a model at a given hook point, last token position.

    Returns: (N, d_model) numpy array.
    """
    model.eval()
    all_acts = []

    with torch.no_grad():
        for start in range(0, inputs.shape[0], chunk_size):
            batch = inputs[start:start + chunk_size].to(device)
            _, cache = model.run_with_cache(batch)
            acts = cache[hook_point][:, -1, :].cpu().numpy()
            all_acts.append(acts)

    return np.concatenate(all_acts, axis=0)


# =============================================================================
# SAE TRAINING
# =============================================================================

def _build_sae(d_model: int, dict_size: int, cfg, device: str) -> SparseAutoencoder:
    """Build an SAE with the configured activation function."""
    return SparseAutoencoder(
        d_model=d_model,
        dict_size=dict_size,
        activation=cfg.activation,
        k=cfg.topk_k,
        jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        jumprelu_init_threshold=cfg.jumprelu_init_threshold,
    ).to(device)


def _train_sae(
    activations: np.ndarray,
    cfg,
    device: str,
    seed: int = 42,
) -> tuple[SparseAutoencoder, SAEStats]:
    """Train SAE on activation vectors.

    Args:
        activations: (N, d_model) numpy array
        cfg: SAEConfig
        device: torch device string
    """
    d_model = activations.shape[1]
    dict_size = cfg.dict_size_multiplier * d_model

    # Validate matryoshka config
    matryoshka_widths = cfg.matryoshka_widths if cfg.matryoshka_widths else None
    if matryoshka_widths:
        assert cfg.activation == "batch_topk", \
            f"Matryoshka training requires batch_topk activation, got {cfg.activation}"
        if matryoshka_widths[-1] != dict_size:
            matryoshka_widths = matryoshka_widths + [dict_size]
        assert matryoshka_widths == sorted(matryoshka_widths), \
            "Matryoshka widths must be in ascending order"
        assert all(w > 0 for w in matryoshka_widths), \
            "All matryoshka widths must be positive"
        print(f"  Matryoshka training: widths = {matryoshka_widths}, inner_weight = {cfg.matryoshka_inner_weight}")

    sae = _build_sae(d_model, dict_size, cfg, device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.learning_rate)

    act_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    n_total = act_tensor.shape[0]
    rng = np.random.default_rng(seed)

    loss_curve = []
    final_recon = 0.0
    final_sparsity = 0.0
    final_per_width_recon = None

    # Dead feature tracking: count steps since each feature last fired
    dead_window = cfg.dead_window
    steps_since_active = torch.zeros(dict_size, dtype=torch.long, device=device)

    for step in tqdm.tqdm(range(1, cfg.num_steps + 1), desc="Training SAE"):
        idx = rng.choice(n_total, size=min(cfg.batch_size, n_total), replace=True)
        batch = act_tensor[idx]

        # Build dead mask from activity tracker
        dead_mask = steps_since_active >= dead_window

        loss, recon_loss, sparsity_loss, z, per_width_recon = sae.compute_loss(
            batch, l1_coefficient=cfg.l1_coefficient, dead_mask=dead_mask,
            aux_coefficient=cfg.aux_coefficient,
            matryoshka_widths=matryoshka_widths,
            matryoshka_inner_weight=cfg.matryoshka_inner_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize decoder for relu/jumprelu; topk/batch_topk rely on
        # the aux loss to keep decoder norms in check, but normalizing
        # doesn't hurt and keeps things consistent.
        sae.normalize_decoder()

        # Update dead feature tracker
        with torch.no_grad():
            fired = (z > 0).any(dim=0)  # (dict_size,)
            steps_since_active += 1
            steps_since_active[fired] = 0

        loss_curve.append(loss.item())
        final_recon = recon_loss.item()
        final_sparsity = sparsity_loss.item()
        if per_width_recon is not None:
            final_per_width_recon = per_width_recon

    # Compute eval diagnostics on full dataset (in chunks)
    sae.eval()
    all_l0 = []
    feature_ever_active = np.zeros(dict_size, dtype=bool)

    with torch.no_grad():
        for start in range(0, n_total, cfg.batch_size):
            batch = act_tensor[start:start + cfg.batch_size]
            z = sae.encode(batch)
            active = (z > 0).cpu().numpy()
            all_l0.extend(active.sum(axis=1).tolist())
            feature_ever_active |= active.any(axis=0)

    if final_per_width_recon:
        print(f"  Matryoshka per-width recon losses: " +
              ", ".join(f"w={w}: {l:.4f}" for w, l in sorted(final_per_width_recon.items())))

    sae_stats = SAEStats(
        final_loss=loss_curve[-1],
        final_reconstruction_loss=final_recon,
        final_l1_loss=final_sparsity,
        mean_l0=float(np.mean(all_l0)),
        dead_features=int((~feature_ever_active).sum()),
        loss_curve=loss_curve,
        matryoshka_recon_losses=final_per_width_recon,
    )

    return sae, sae_stats


# =============================================================================
# STANDALONE SAE TRAINING
# =============================================================================

def train_sae_standalone(
    run_dir: str | Path,
    sae_cfg: "SAEConfig | None" = None,
    device: str = "auto",
    seed: int = 42,
    save_dir: str | Path | None = None,
) -> tuple[SparseAutoencoder, SAEStats, np.ndarray]:
    """Train an SAE in isolation using a previous pipeline run's model.

    Loads process + pretrained model from run_dir, generates sequences,
    extracts activations, and trains an SAE. Useful for quickly iterating
    on SAE hyperparameters without running the full diffing pipeline.

    Args:
        run_dir: Path to a pipeline run directory (e.g. outputs/run_20260303_0652).
        sae_cfg: SAE configuration. If None, uses default SAEConfig().
        device: Torch device ("auto", "cpu", "cuda").
        seed: Random seed.
        save_dir: If provided, saves SAE weights and training activations here.

    Returns:
        (sae, sae_stats, activations) tuple.
    """
    from em_pipeline.config import SAEConfig
    from em_pipeline import process as process_mod
    from em_pipeline import pretrain as pretrain_mod

    if sae_cfg is None:
        sae_cfg = SAEConfig()

    run_dir = Path(run_dir)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load previous run artifacts
    print(f"Loading process from {run_dir / 'stage1'}...")
    process = process_mod.load(run_dir / "stage1")

    print(f"Loading pretrained model from {run_dir / 'stage3'}...")
    pretrain = pretrain_mod.load(run_dir / "stage3")

    total_vocab = process.info["total_vocab"]
    hook_point = _resolve_hook_point(pretrain)
    print(f"Hook point: {hook_point}")

    # Generate activations
    print(f"\nGenerating {sae_cfg.n_train_samples} sequences for SAE training...")
    base_model = _load_base_model(pretrain, total_vocab, device)
    train_inputs, _ = _generate_eval_sequences(
        process, sae_cfg.n_train_samples, seed=seed + 5000, device=device,
    )
    activations = _extract_activations(base_model, train_inputs, hook_point, device)
    del base_model
    print(f"  Activations shape: {activations.shape}")

    # Train SAE
    print(f"\nTraining SAE (dict_size={sae_cfg.dict_size_multiplier}x{activations.shape[1]}, "
          f"activation={sae_cfg.activation}, k={sae_cfg.topk_k}, "
          f"steps={sae_cfg.num_steps})...")
    sae, sae_stats = _train_sae(activations, sae_cfg, device, seed=seed)

    # Print stats
    sparsity_label = "L1" if sae_cfg.activation == "relu" else "aux"
    print(f"\n  Final loss: {sae_stats.final_loss:.4f} "
          f"(recon={sae_stats.final_reconstruction_loss:.4f}, "
          f"{sparsity_label}={sae_stats.final_l1_loss:.4f})")
    print(f"  Mean L0: {sae_stats.mean_l0:.1f}, "
          f"Dead features: {sae_stats.dead_features}/{sae.dict_size}")

    # Optionally save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), save_dir / "sae.pt")
        np.save(save_dir / "train_activations.npy", activations)
        save_pickle(sae_stats, save_dir / "sae_stats.pkl")
        print(f"\n  Saved to {save_dir}/")

    return sae, sae_stats, activations


# =============================================================================
# FEATURE DIFFING
# =============================================================================

def _compute_feature_diffs(
    sae: nn.Module,
    base_act: np.ndarray,
    ft_a_act: np.ndarray,
    ft_b_act: np.ndarray,
    base_beliefs: np.ndarray,
    process: ProcessResult,
    top_k: int,
    device: str,
) -> tuple[list[FeatureDiffStats], list[FeatureDiffStats], list[FeatureDiffStats],
           np.ndarray, np.ndarray, np.ndarray]:
    """Project activations through SAE and compute per-feature diffs.

    Returns (all_features, top_features_a, top_features_b, z_base, z_ft_a, z_ft_b).
    """
    info = process.info
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])

    sae.eval()
    with torch.no_grad():
        z_base = sae.encode(torch.tensor(base_act, dtype=torch.float32, device=device)).cpu().numpy()
        z_ft_a = sae.encode(torch.tensor(ft_a_act, dtype=torch.float32, device=device)).cpu().numpy()
        z_ft_b = sae.encode(torch.tensor(ft_b_act, dtype=torch.float32, device=device)).cpu().numpy()

    mean_base = z_base.mean(axis=0)
    mean_ft_a = z_ft_a.mean(axis=0)
    mean_ft_b = z_ft_b.mean(axis=0)

    # Compute belief correlations for base activations
    pi_a = base_beliefs[:, sector_a_idx].sum(axis=1)

    # Within-sector normalized beliefs (use first component as summary)
    raw_a = base_beliefs[:, sector_a_idx]
    mass_a = raw_a.sum(axis=1)
    valid_a = mass_a > 0.01
    mu_a_0 = np.where(valid_a, raw_a[:, 0] / np.maximum(mass_a, 1e-8), np.nan)

    raw_b = base_beliefs[:, sector_b_idx]
    mass_b = raw_b.sum(axis=1)
    valid_b = mass_b > 0.01
    mu_b_0 = np.where(valid_b, raw_b[:, 0] / np.maximum(mass_b, 1e-8), np.nan)

    dict_size = z_base.shape[1]
    all_features = []

    for j in range(dict_size):
        z_j = z_base[:, j]

        # Pearson correlation with pi_A
        corr_pi_a = None
        if z_j.std() > 1e-8 and pi_a.std() > 1e-8:
            corr_pi_a = float(stats.pearsonr(z_j, pi_a)[0])

        # Correlation with within-sector beliefs
        corr_within_a = None
        valid_mask_a = valid_a & ~np.isnan(mu_a_0)
        if valid_mask_a.sum() > 10 and z_j[valid_mask_a].std() > 1e-8:
            corr_within_a = float(stats.pearsonr(z_j[valid_mask_a], mu_a_0[valid_mask_a])[0])

        corr_within_b = None
        valid_mask_b = valid_b & ~np.isnan(mu_b_0)
        if valid_mask_b.sum() > 10 and z_j[valid_mask_b].std() > 1e-8:
            corr_within_b = float(stats.pearsonr(z_j[valid_mask_b], mu_b_0[valid_mask_b])[0])

        all_features.append(FeatureDiffStats(
            feature_idx=j,
            mean_base=float(mean_base[j]),
            mean_ft_a=float(mean_ft_a[j]),
            mean_ft_b=float(mean_ft_b[j]),
            diff_a=float(mean_ft_a[j] - mean_base[j]),
            diff_b=float(mean_ft_b[j] - mean_base[j]),
            corr_pi_a=corr_pi_a,
            corr_within_a=corr_within_a,
            corr_within_b=corr_within_b,
        ))

    # Sort for top features
    top_a = sorted(all_features, key=lambda f: f.diff_a, reverse=True)[:top_k]
    top_b = sorted(all_features, key=lambda f: f.diff_b, reverse=True)[:top_k]

    return all_features, top_a, top_b, z_base, z_ft_a, z_ft_b


# =============================================================================
# MODEL LOADING HELPERS
# =============================================================================

def _load_base_model(pretrain: PretrainResult, total_vocab: int, device: str):
    """Load the pretrained base model (reuses finetune._load_base_model logic)."""
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


def _load_ft_model(
    pretrain: PretrainResult,
    total_vocab: int,
    sector_dir: Path,
    ft_checkpoint: str,
    device: str,
):
    """Load a finetuned model from a sector checkpoint directory."""
    model = _load_base_model(pretrain, total_vocab, device)

    if ft_checkpoint == "final":
        # Find highest-numbered checkpoint (sort numerically, not lexicographically)
        ckpt_files = list(sector_dir.glob("ft_step_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No FT checkpoints found in {sector_dir}")
        ckpt_path = max(ckpt_files, key=lambda p: int(p.stem.split("_")[-1]))
    else:
        ckpt_path = sector_dir / f"ft_step_{ft_checkpoint}.pt"

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    print(f"  Loaded FT checkpoint: {ckpt_path.name}")
    return model


def _resolve_hook_point(pretrain: PretrainResult) -> str:
    """Resolve 'auto' hook point to last layer resid_post."""
    stage_dir = Path(pretrain.run_dir)
    model_config = load_json(stage_dir / "model_config.json")
    n_layers = model_config["train_config"]["n_layers"]
    return f"blocks.{n_layers - 1}.hook_resid_post"


# =============================================================================
# CAUSAL STEERING VALIDATION
# =============================================================================

def _generate_prompt_sequences(
    process: ProcessResult,
    n_samples: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Generate prompt-only sequences for steering evaluation.

    Returns: (N, prompt_len) tensor of prompt tokens (including BOS).
    """
    from training.matrices import generate_afp_batch

    info = process.info
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]

    device_arg = torch.device(device) if device != "cpu" else None
    key = jax.random.key(seed)
    inputs, _ = generate_afp_batch(
        process.prompt_hmm, process.comp_hmm,
        batch_size=n_samples,
        prompt_len=prompt_len,
        comp_len=comp_len,
        key=key,
        v_p=v_p,
        device=device_arg,
    )
    # inputs is (N, seq_len-1) = (N, prompt_len + comp_len - 1)
    # We want just the prompt portion: first prompt_len columns
    return inputs[:, :prompt_len].to(device)


def _generate_completions_steered(
    model: torch.nn.Module,
    prompt_tensor: torch.Tensor,
    v_p: int,
    comp_len: int,
    total_vocab: int,
    n_completions: int,
    device: str,
    fwd_hooks: list | None = None,
    chunk_size: int = 64,
    temperature: float = 1.0,
) -> np.ndarray:
    """Autoregressively generate completions, optionally with steering hooks.

    Like finetune._generate_completions but accepts a prompt tensor directly
    and applies fwd_hooks on every forward pass so steering fires during
    the entire autoregressive loop.

    Returns: (N, n_completions, comp_len) array of token indices.
    """
    model.eval()
    n_prompts = prompt_tensor.shape[0]
    result = np.zeros((n_prompts, n_completions, comp_len), dtype=np.int64)

    logit_mask = torch.zeros(total_vocab, device=device)
    logit_mask[:v_p] = float("-inf")

    ctx = model.hooks(fwd_hooks=fwd_hooks) if fwd_hooks else _nullcontext()

    with torch.no_grad(), ctx:
        for p_start in range(0, n_prompts, chunk_size):
            p_end = min(p_start + chunk_size, n_prompts)
            chunk = prompt_tensor[p_start:p_end]  # (n_chunk, prompt_len)
            n_chunk = p_end - p_start

            # Repeat each prompt n_completions times: (n_chunk * n_completions, prompt_len)
            batch = chunk.repeat_interleave(n_completions, dim=0)

            generated_tokens = []
            current = batch
            for _step in range(comp_len):
                outputs = model(current)
                last_logits = outputs[:, -1, :] + logit_mask
                probs = torch.softmax(last_logits / temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(sampled)
                current = torch.cat([current, sampled], dim=1)

            gen = torch.cat(generated_tokens, dim=1).cpu().numpy()
            gen = gen.reshape(n_chunk, n_completions, comp_len)
            result[p_start:p_end] = gen

    return result


class _nullcontext:
    """Minimal no-op context manager (for Python < 3.10 compat)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def _tensor_to_prompt_keys(prompt_tensor: torch.Tensor) -> list[str]:
    """Convert (N, prompt_len) tensor to list of string keys like '(0, 1, 2)'.

    Compatible with finetune._score_generated_completions prompt_keys format.
    """
    return [str(tuple(row.tolist())) for row in prompt_tensor.cpu()]


def _score_generated_belief_consistency(
    generated: np.ndarray,       # (n_prompts, n_completions, comp_len)
    prompt_keys: list[str],
    process: ProcessResult,
    v_p: int,
    prompt_len: int,
) -> dict:
    """Compute belief consistency metrics for generated completions.

    Returns dict with:
      - "mean_final_pi_a": float — same as _score_generated_completions
      - "mean_sector_commitment": float — fraction of positions with π_A > 0.8 or < 0.2
      - "mean_trajectory_monotonicity": float — abs(Spearman corr) of π_A trajectory
      - "mean_sector_confidence": float — mean of max(π_A, 1-π_A) across positions
    """
    from em_pipeline.finetune import _parse_prompt_key

    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    T_comp = np.array(process.comp_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)
    sector_a_idx = to_np_idx(process.info["sector_a_idx"])

    n_prompts, n_completions, comp_len = generated.shape

    all_final_pi_a = []
    all_commitment = []
    all_monotonicity = []
    all_confidence = []

    for p_idx in range(n_prompts):
        prompt_seq = _parse_prompt_key(prompt_keys[p_idx])

        # Compute post-prompt belief state
        post_prompt = init.copy()
        for tok in prompt_seq:
            post_prompt = post_prompt @ T_prompt[tok]
            s = post_prompt.sum()
            if s > 0:
                post_prompt /= s

        for c_idx in range(n_completions):
            pi_a_trajectory = []
            state = post_prompt.copy()
            for t in range(comp_len):
                tok = int(generated[p_idx, c_idx, t])
                comp_tok = tok - v_p
                if 0 <= comp_tok < T_comp.shape[0]:
                    state = state @ T_comp[comp_tok]
                    s = state.sum()
                    if s > 0:
                        state /= s
                pi_a_trajectory.append(float(state[sector_a_idx].sum()))

            traj = np.array(pi_a_trajectory)

            # Final π_A
            all_final_pi_a.append(traj[-1])

            # Sector commitment: fraction of positions with decisive belief
            commitment = np.mean((traj > 0.8) | (traj < 0.2))
            all_commitment.append(commitment)

            # Trajectory monotonicity: abs(Spearman corr with [0,1,...,T])
            if len(traj) > 2 and np.std(traj) > 1e-10:
                corr, _ = stats.spearmanr(traj, np.arange(len(traj)))
                all_monotonicity.append(abs(corr))
            else:
                # Constant trajectory = perfectly monotone (already committed)
                all_monotonicity.append(1.0)

            # Sector confidence: mean of max(π_A, 1-π_A)
            confidence = np.mean(np.maximum(traj, 1.0 - traj))
            all_confidence.append(confidence)

    return {
        "mean_final_pi_a": float(np.mean(all_final_pi_a)),
        "mean_sector_commitment": float(np.mean(all_commitment)),
        "mean_trajectory_monotonicity": float(np.mean(all_monotonicity)),
        "mean_sector_confidence": float(np.mean(all_confidence)),
    }


def _run_generative_steering(
    model: torch.nn.Module,
    hook_point: str,
    decoder_vec: np.ndarray,
    scales: list[float],
    gen_prompts: torch.Tensor,
    process: ProcessResult,
    steer_cfg,
    device: str,
) -> tuple[list[float], list[dict]]:
    """Run generative steering for one feature across all scales.

    Returns (gen_pi_a_per_scale, belief_consistency_per_scale).
    """
    from em_pipeline.finetune import _score_generated_completions

    info = process.info
    v_p = info["v_p"]
    comp_len = info["comp_len"]
    total_vocab = info["total_vocab"]
    prompt_len = info["prompt_len"]
    n_completions = steer_cfg.n_gen_completions

    prompt_keys = _tensor_to_prompt_keys(gen_prompts)

    # Cache unsteered generation for scale=0
    unsteered_gen = None
    gen_pi_a_per_scale = []
    belief_consistency_per_scale = []

    for scale in scales:
        if scale == 0.0:
            generated = _generate_completions_steered(
                model, gen_prompts, v_p, comp_len, total_vocab,
                n_completions, device,
            )
            unsteered_gen = generated
        else:
            steering_tensor = torch.tensor(
                decoder_vec * scale, dtype=torch.float32, device=device,
            )

            def hook(activation, *, hook, _st=steering_tensor, _pos=steer_cfg.position):
                if _pos == "all":
                    activation += _st
                elif _pos == "last":
                    activation[:, -1, :] += _st
                return activation

            generated = _generate_completions_steered(
                model, gen_prompts, v_p, comp_len, total_vocab,
                n_completions, device, fwd_hooks=[(hook_point, hook)],
            )

        per_prompt_pi_a = _score_generated_completions(
            generated, prompt_keys, process, v_p, prompt_len,
        )
        gen_pi_a_per_scale.append(float(np.mean(per_prompt_pi_a)))

        consistency = _score_generated_belief_consistency(
            generated, prompt_keys, process, v_p, prompt_len,
        )
        belief_consistency_per_scale.append(consistency)

    return gen_pi_a_per_scale, belief_consistency_per_scale


def run_steering_validation(
    base_model: torch.nn.Module,
    process: ProcessResult,
    diffing_result: DiffingResult,
    cfg: DiffingConfig,
    pipeline_cfg: PipelineConfig,
    device: str,
) -> SteeringResult:
    """Causally validate top SAE features by steering.

    For each top feature, adds scaled decoder vectors to the residual stream
    and measures how P(A-tagged) / P(B-tagged) change.
    """
    info = process.info
    v_p = info["v_p"]
    total_vocab = info["total_vocab"]
    v_c = (total_vocab - v_p) // 2

    steer_cfg = cfg.steering
    hook_point = diffing_result.hook_point

    # Generate prompts for evaluation
    print(f"\nGenerating steering eval prompts ({steer_cfg.n_eval_prompts})...")
    prompts = _generate_prompt_sequences(
        process, steer_cfg.n_eval_prompts,
        seed=pipeline_cfg.seed + 7000, device=device,
    )

    # Collect features to steer: top_k from A and top_k from B (deduplicated)
    top_k = steer_cfg.top_k_steer
    features_to_steer = []
    seen_idx = set()

    for i, f in enumerate(diffing_result.top_features_a[:top_k]):
        if f.feature_idx not in seen_idx:
            vec = np.array(diffing_result.decoder_vectors_a[i])
            features_to_steer.append((f.feature_idx, vec, "A"))
            seen_idx.add(f.feature_idx)

    for i, f in enumerate(diffing_result.top_features_b[:top_k]):
        if f.feature_idx not in seen_idx:
            vec = np.array(diffing_result.decoder_vectors_b[i])
            features_to_steer.append((f.feature_idx, vec, "B"))
            seen_idx.add(f.feature_idx)

    print(f"Steering {len(features_to_steer)} features at {len(steer_cfg.scales)} scales...")
    coherence_thresh = steer_cfg.coherence_threshold

    # A-tagged and B-tagged token slices
    a_slice = slice(v_p, v_p + v_c)
    b_slice = slice(v_p + v_c, v_p + 2 * v_c)

    # Get baseline (unsteered) distribution for coherence reference
    base_model.eval()
    with torch.no_grad():
        base_logits = base_model(prompts)
    base_probs = torch.softmax(base_logits[:, -1, :], dim=-1)  # (N, vocab)
    # Within-sector conditional distributions (per prompt)
    base_a_mass = base_probs[:, a_slice].sum(dim=-1, keepdim=True).clamp(min=1e-10)
    base_b_mass = base_probs[:, b_slice].sum(dim=-1, keepdim=True).clamp(min=1e-10)
    base_cond_a = base_probs[:, a_slice] / base_a_mass  # (N, v_c)
    base_cond_b = base_probs[:, b_slice] / base_b_mass  # (N, v_c)

    per_feature_results = []

    for feat_idx, decoder_vec, source in tqdm.tqdm(features_to_steer, desc="Steering features"):
        p_a_per_scale = []
        p_b_per_scale = []
        coherence_per_scale = []
        kl_loss_per_scale = []

        for scale in steer_cfg.scales:
            if scale == 0.0:
                # Use cached baseline
                probs = base_probs
            else:
                steering_tensor = torch.tensor(
                    decoder_vec * scale,
                    dtype=torch.float32,
                    device=device,
                )

                def hook(activation, *, hook, _st=steering_tensor, _pos=steer_cfg.position):
                    if _pos == "all":
                        activation += _st
                    elif _pos == "last":
                        activation[:, -1, :] += _st
                    return activation

                with torch.no_grad():
                    with base_model.hooks(fwd_hooks=[(hook_point, hook)]):
                        logits = base_model(prompts)

                probs = torch.softmax(logits[:, -1, :], dim=-1)

            # P(A-tagged) and P(B-tagged)
            steer_a_mass = probs[:, a_slice].sum(dim=-1)  # (N,)
            steer_b_mass = probs[:, b_slice].sum(dim=-1)  # (N,)
            p_a = steer_a_mass.mean().item()
            p_b = steer_b_mass.mean().item()

            # Coherence: prob-weighted within-sector KL divergence from base
            # KL(steer_cond_A || base_cond_A) per prompt, weighted by P_steer(A)
            # KL(steer_cond_B || base_cond_B) per prompt, weighted by P_steer(B)
            steer_cond_a = probs[:, a_slice] / steer_a_mass.unsqueeze(-1).clamp(min=1e-10)
            steer_cond_b = probs[:, b_slice] / steer_b_mass.unsqueeze(-1).clamp(min=1e-10)

            eps = 1e-10
            kl_a = (steer_cond_a * (torch.log(steer_cond_a + eps) - torch.log(base_cond_a + eps))).sum(dim=-1)
            kl_b = (steer_cond_b * (torch.log(steer_cond_b + eps) - torch.log(base_cond_b + eps))).sum(dim=-1)

            # Weighted coherence loss per prompt, then average
            coh_loss = (steer_a_mass * kl_a + steer_b_mass * kl_b).mean().item()
            # Convert to coherence score in [0, 1]: exp(-loss)
            coh = float(np.exp(-coh_loss))

            p_a_per_scale.append(p_a)
            p_b_per_scale.append(p_b)
            coherence_per_scale.append(coh)
            kl_loss_per_scale.append(coh_loss)

        # Find baseline index (scale=0)
        baseline_idx = None
        for i, s in enumerate(steer_cfg.scales):
            if s == 0.0:
                baseline_idx = i
                break
        mid = baseline_idx if baseline_idx is not None else len(steer_cfg.scales) // 2
        p_a_baseline = p_a_per_scale[mid]
        coh_baseline = coherence_per_scale[mid]

        # Adaptive scaling: find largest pos/neg scale with coherence ≥ threshold
        adapted_pos = None
        p_a_adapted_pos = None
        adapted_neg = None
        p_a_adapted_neg = None

        for i, s in enumerate(steer_cfg.scales):
            if s > 0 and coherence_per_scale[i] >= coherence_thresh:
                if adapted_pos is None or s > adapted_pos:
                    adapted_pos = s
                    p_a_adapted_pos = p_a_per_scale[i]
            if s < 0 and coherence_per_scale[i] >= coherence_thresh:
                if adapted_neg is None or s < adapted_neg:
                    adapted_neg = s
                    p_a_adapted_neg = p_a_per_scale[i]

        per_feature_results.append(FeatureSteeringResult(
            feature_idx=feat_idx,
            scales=list(steer_cfg.scales),
            p_a_tagged=p_a_per_scale,
            p_b_tagged=p_b_per_scale,
            coherence=coherence_per_scale,
            kl_loss=kl_loss_per_scale,
            p_a_baseline=p_a_baseline,
            coherence_baseline=coh_baseline,
            adapted_pos_scale=adapted_pos,
            adapted_neg_scale=adapted_neg,
            p_a_at_adapted_pos=p_a_adapted_pos,
            p_a_at_adapted_neg=p_a_adapted_neg,
        ))

        adapted_str = f"adapted=[{adapted_neg}, {adapted_pos}]"
        print(f"  Feature {feat_idx} (top-{source}): "
              f"P(A) range [{min(p_a_per_scale):.3f}, {max(p_a_per_scale):.3f}], "
              f"coh range [{min(coherence_per_scale):.3f}, {max(coherence_per_scale):.3f}], "
              f"{adapted_str}")

    # ── Generative steering evaluation ──────────────────────────
    if steer_cfg.n_gen_prompts > 0:
        print(f"\nGenerative steering eval ({steer_cfg.n_gen_prompts} prompts, "
              f"{steer_cfg.n_gen_completions} completions/prompt)...")
        gen_prompts = _generate_prompt_sequences(
            process, steer_cfg.n_gen_prompts,
            seed=pipeline_cfg.seed + 8000, device=device,
        )

        for i, (feat_idx, decoder_vec, source) in enumerate(features_to_steer):
            gen_pi_a, belief_consistency = _run_generative_steering(
                base_model, hook_point, decoder_vec,
                list(steer_cfg.scales), gen_prompts, process,
                steer_cfg, device,
            )
            per_feature_results[i].gen_pi_a = gen_pi_a
            per_feature_results[i].gen_belief_consistency = belief_consistency

            gen_range_str = f"gen π_A [{min(gen_pi_a):.3f}, {max(gen_pi_a):.3f}]"
            # Show commitment at scale=0 vs extreme scales
            zero_idx = list(steer_cfg.scales).index(0.0) if 0.0 in steer_cfg.scales else len(steer_cfg.scales) // 2
            commit_0 = belief_consistency[zero_idx]["mean_sector_commitment"]
            commit_range = [bc["mean_sector_commitment"] for bc in belief_consistency]
            print(f"  Feature {feat_idx} (top-{source}): {gen_range_str}, "
                  f"commitment@0={commit_0:.3f}, range=[{min(commit_range):.3f}, {max(commit_range):.3f}]")

    return SteeringResult(
        per_feature=per_feature_results,
        config=steer_cfg,
    )


# =============================================================================
# MAIN RUN
# =============================================================================

def run(
    process: ProcessResult,
    analysis: AnalysisResult,
    pretrain: PretrainResult,
    finetune: FinetuneResult,
    cfg: DiffingConfig,
    pipeline_cfg: PipelineConfig,
) -> DiffingResult:
    """Run SAE-based model diffing."""
    info = process.info
    total_vocab = info["total_vocab"]
    prompt_len = info["prompt_len"]

    device = pipeline_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve hook point
    hook_point = cfg.hook_point
    if hook_point == "auto":
        hook_point = _resolve_hook_point(pretrain)
    print(f"Hook point: {hook_point}")

    # Output directory
    output_dir = Path(pipeline_cfg.output_dir)
    if pipeline_cfg.run_name:
        output_dir = output_dir / pipeline_cfg.run_name
    stage_dir = output_dir / "stage5"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Collect training activations from base model ---
    print(f"\nCollecting SAE training activations ({cfg.sae.n_train_samples} samples)...")
    base_model = _load_base_model(pretrain, total_vocab, device)

    train_inputs, _ = _generate_eval_sequences(
        process, cfg.sae.n_train_samples,
        seed=pipeline_cfg.seed + 5000, device=device,
    )
    train_activations = _extract_activations(base_model, train_inputs, hook_point, device)
    print(f"  Training activations shape: {train_activations.shape}")

    # --- Step 2: Train SAE ---
    print(f"\nTraining SAE (dict_size={cfg.sae.dict_size_multiplier}x{train_activations.shape[1]}, "
          f"activation={cfg.sae.activation}, k={cfg.sae.topk_k})...")
    sae, sae_stats = _train_sae(
        train_activations, cfg.sae, device, seed=pipeline_cfg.seed,
    )
    sparsity_label = "L1" if cfg.sae.activation == "relu" else "aux"
    print(f"  Final loss: {sae_stats.final_loss:.4f} "
          f"(recon={sae_stats.final_reconstruction_loss:.4f}, "
          f"{sparsity_label}={sae_stats.final_l1_loss:.4f})")
    print(f"  Mean L0: {sae_stats.mean_l0:.1f}, "
          f"Dead features: {sae_stats.dead_features}/{sae.dict_size}")

    # Save SAE weights
    torch.save(sae.state_dict(), stage_dir / "sae.pt")

    # --- Step 3: Generate shared eval sequences ---
    print(f"\nGenerating eval sequences ({cfg.n_eval_samples} samples)...")
    eval_inputs, eval_beliefs = _generate_eval_sequences(
        process, cfg.n_eval_samples,
        seed=pipeline_cfg.seed + 6000, device=device,
    )

    # Beliefs at last input position (for interpretation)
    last_pos = eval_inputs.shape[1] - 1
    beliefs_at_last = eval_beliefs[:, last_pos, :]

    # --- Step 4: Collect activations from all 3 models ---
    print("\nCollecting base model eval activations...")
    base_eval_act = _extract_activations(base_model, eval_inputs, hook_point, device)
    # Keep base_model alive for steering validation below

    # Find sector checkpoint directories (derive from pretrain.run_dir which
    # correctly points to the loaded location, even across --load-from runs)
    ft_dir = Path(pretrain.run_dir).parent / "stage4"
    sector_a_dir = ft_dir / "sector_a"
    sector_b_dir = ft_dir / "sector_b"

    print("Collecting FT-A model eval activations...")
    ft_a_model = _load_ft_model(pretrain, total_vocab, sector_a_dir, cfg.ft_checkpoint, device)
    ft_a_eval_act = _extract_activations(ft_a_model, eval_inputs, hook_point, device)
    del ft_a_model

    print("Collecting FT-B model eval activations...")
    ft_b_model = _load_ft_model(pretrain, total_vocab, sector_b_dir, cfg.ft_checkpoint, device)
    ft_b_eval_act = _extract_activations(ft_b_model, eval_inputs, hook_point, device)
    del ft_b_model

    # --- Step 5: Compute feature diffs ---
    print(f"\nComputing feature diffs (top {cfg.top_k_features})...")
    all_features, top_a, top_b, z_base, z_ft_a, z_ft_b = _compute_feature_diffs(
        sae, base_eval_act, ft_a_eval_act, ft_b_eval_act,
        beliefs_at_last, process, cfg.top_k_features, device,
    )

    # Extract decoder vectors for top features
    # W_dec: (dict_size, d_model) — each row is a decoder vector
    W_dec = sae.W_dec.detach().cpu().numpy()
    decoder_vectors_a = [W_dec[f.feature_idx, :].tolist() for f in top_a]
    decoder_vectors_b = [W_dec[f.feature_idx, :].tolist() for f in top_b]

    # Print summary
    print(f"\nTop {cfg.top_k_features} features for sector A finetuning:")
    for f in top_a[:10]:
        corr_str = f"corr_pi_a={f.corr_pi_a:.3f}" if f.corr_pi_a is not None else "corr_pi_a=N/A"
        print(f"  Feature {f.feature_idx}: diff_a={f.diff_a:+.4f}, diff_b={f.diff_b:+.4f}, {corr_str}")

    print(f"\nTop {cfg.top_k_features} features for sector B finetuning:")
    for f in top_b[:10]:
        corr_str = f"corr_pi_a={f.corr_pi_a:.3f}" if f.corr_pi_a is not None else "corr_pi_a=N/A"
        print(f"  Feature {f.feature_idx}: diff_a={f.diff_a:+.4f}, diff_b={f.diff_b:+.4f}, {corr_str}")

    sector_a_idx = to_np_idx(process.info["sector_a_idx"])

    # --- Step 6: Causal steering validation ---
    print("\n--- Causal Steering Validation ---")
    # Build temporary result for steering (needs decoder vectors + hook point)
    temp_result = DiffingResult(
        sae_stats=sae_stats,
        hook_point=hook_point,
        all_features=all_features,
        top_features_a=top_a,
        top_features_b=top_b,
        decoder_vectors_a=decoder_vectors_a,
        decoder_vectors_b=decoder_vectors_b,
    )
    steering_result = run_steering_validation(
        base_model, process, temp_result, cfg, pipeline_cfg, device,
    )
    del base_model  # free memory

    # --- Step 7: Post-finetune steering validation ---
    print("\n--- Post-Finetune Steering Validation ---")

    print("Steering FT-A model...")
    ft_a_model = _load_ft_model(pretrain, total_vocab, sector_a_dir, cfg.ft_checkpoint, device)
    steering_result_ft_a = run_steering_validation(
        ft_a_model, process, temp_result, cfg, pipeline_cfg, device,
    )
    del ft_a_model

    print("Steering FT-B model...")
    ft_b_model = _load_ft_model(pretrain, total_vocab, sector_b_dir, cfg.ft_checkpoint, device)
    steering_result_ft_b = run_steering_validation(
        ft_b_model, process, temp_result, cfg, pipeline_cfg, device,
    )
    del ft_b_model

    return DiffingResult(
        sae_stats=sae_stats,
        hook_point=hook_point,
        all_features=all_features,
        top_features_a=top_a,
        top_features_b=top_b,
        decoder_vectors_a=decoder_vectors_a,
        decoder_vectors_b=decoder_vectors_b,
        eval_z_base=z_base,
        eval_z_ft_a=z_ft_a,
        eval_z_ft_b=z_ft_b,
        eval_beliefs=beliefs_at_last,
        eval_sector_a_idx=sector_a_idx.tolist(),
        steering_result=steering_result,
        steering_result_ft_a=steering_result_ft_a,
        steering_result_ft_b=steering_result_ft_b,
    )


def save(result: DiffingResult, path: Path) -> None:
    """Save diffing results to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(result, path / "diffing_result.pkl")


def load(path: Path) -> DiffingResult:
    """Load diffing results from disk."""
    return load_pickle(Path(path) / "diffing_result.pkl")


# =============================================================================
# FINETUNE RESULT EXTRACTION HELPER
# =============================================================================

def extract_finetune_result(finetune) -> FinetuneResult | None:
    """Extract a FinetuneResult from whatever Stage 4 produced."""
    if isinstance(finetune, FinetuneResult):
        return finetune
    elif isinstance(finetune, CorrectionSweepResult):
        return finetune.per_epsilon.get(0.0) or next(iter(finetune.per_epsilon.values()))
    elif isinstance(finetune, CorrectionMixResult):
        return next(iter(finetune.per_frac.values()))
    return None
