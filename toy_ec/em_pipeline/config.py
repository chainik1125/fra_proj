"""Configuration and result dataclasses for the AFP steering pipeline."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import json
import pickle
import yaml
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ProcessConfig:
    """AFP generative process parameters."""
    variant: str = "z1r_afp"   # "z1r_afp" | "metastable4" | "clustered_codebook" | "leaky_reset"
    delta: float = 0.05        # Z1R' leak parameter
    beta: float = 0.556        # completion collapse rate
    alpha: float = 1.0         # dominant sector scale (1.0 WLOG)
    v_p: int = 10              # prompt vocabulary size
    pi_a: float = 0.5          # initial sector A probability
    bias_range: float = 2.0    # log-ratio range for prompt biases
    # --- leaky_reset-specific (ignored by other variants) ---
    lambda_g: float = 0.6      # G-sector write strength (0=no write, 1=hard reset)
    lambda_b: float = 0.6      # B-sector write strength
    decode_noise: float = 0.05 # completion readout noise δ
    d_g: int = 5               # G-sector hidden states
    d_b: int = 5               # B-sector hidden states
    content_symbols: int = 5   # M: number of content indices for completion
    signature_type: str = "onehot"  # "onehot" | "spread"


@dataclass
class SequenceConfig:
    """Sequence structure parameters."""
    prompt_len: int = 3        # prompt length
    comp_len: int = 5          # completion length


@dataclass
class ModelConfig:
    """Transformer architecture parameters."""
    d_model: int = 64
    d_head: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_mlp: int = 256


@dataclass
class PretrainConfig:
    """Pretraining hyperparameters."""
    num_steps: int = 20_000
    batch_size: int = 128
    learning_rate: float = 1e-3
    checkpoint_steps: list[int] = field(
        default_factory=lambda: [1000, 5000, 10000, 20000]
    )
    regression_n_samples: int = 500
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class FinetuneConfig:
    """Finetuning hyperparameters."""
    prompt_frac: float = 0.05          # fraction of prompts for finetuning
    ft_steps: int = 2000
    ft_lr: float = 3e-4
    ft_batch_size: int = 64
    sector_threshold: float = 0.9      # rejection sampling threshold
    completions_per_prompt: int = 50
    correction_strengths: list[float] = field(default_factory=lambda: [0.0])
    # Mix-based correction: inject unfiltered corrected sequences into FT data
    correction_mix_fracs: list[float] = field(default_factory=lambda: [0.0])
    correction_mix_epsilon: float = 0.05  # epsilon for the corrected sequences


@dataclass
class SAEConfig:
    """Sparse Autoencoder architecture and training."""
    dict_size_multiplier: int = 4      # SAE hidden dim = multiplier * d_model
    activation: str = "batch_topk"     # "batch_topk", "topk", "relu", "jumprelu"
    topk_k: int = 20                   # k for topk/batch_topk activations
    l1_coefficient: float = 1e-3       # L1 sparsity penalty weight (only used when activation="relu")
    aux_coefficient: float = 1.0 / 32.0  # dead-feature aux loss weight (topk/batch_topk)
    dead_window: int = 200               # steps of inactivity before a feature is "dead"
    jumprelu_bandwidth: float = 0.001    # STE bandwidth ε for JumpReLU threshold learning
    jumprelu_init_threshold: float = 0.01  # initial threshold for JumpReLU features
    learning_rate: float = 1e-3
    num_steps: int = 5000
    batch_size: int = 256              # activation vectors per SAE training step
    n_train_samples: int = 10000       # sequences to generate for SAE training data
    matryoshka_widths: list[int] = field(default_factory=list)  # nested prefix widths, e.g. [32, 64, 128, 256]; empty = standard SAE
    matryoshka_inner_weight: float = 1.0  # weight for inner levels relative to full-width (1.0 = equal, 0.1 = light pressure)


@dataclass
class SteeringConfig:
    """Causal steering validation configuration."""
    scales: list[float] = field(default_factory=lambda: [-100, -50, -20, -10, 0, 10, 20, 50, 100])
    n_eval_prompts: int = 500          # prompts to evaluate per scale
    position: str = "all"              # "all" = add to all token positions (matches paper)
    top_k_steer: int = 5               # how many top features to steer with
    coherence_threshold: float = 0.9   # min P(completion tokens) for adapted scale
    n_gen_prompts: int = 100           # prompts for generative steering eval (expensive)
    n_gen_completions: int = 5         # completions per prompt per scale


@dataclass
class DiffingConfig:
    """SAE-based model-diffing configuration."""
    sae: SAEConfig = field(default_factory=SAEConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    hook_point: str = "auto"           # "auto" = last layer resid_post, or e.g. "blocks.0.hook_resid_post"
    token_position: str = "last"       # "last" = last input token
    n_eval_samples: int = 2000         # sequences for diff evaluation
    top_k_features: int = 20           # top-diffing features to report
    ft_checkpoint: str = "final"       # "final" or step number


@dataclass
class DecompositionConfig:
    """Dimensionality reduction comparison configuration."""
    n_components: int = 10         # Components for PCA/CCA/ICA
    n_samples: int = 2000          # Sequences to generate for analysis
    max_scree_components: int = 20 # Extended PCA scree analysis


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    output_dir: str = "analysis/em_pipeline/outputs"
    run_name: str | None = None  # auto: run_YYYYMMDD_HHMM
    process: ProcessConfig = field(default_factory=ProcessConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    diffing: DiffingConfig = field(default_factory=DiffingConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return _dict_to_config(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _config_to_dict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _dict_to_config(data: dict) -> PipelineConfig:
    """Recursively build PipelineConfig from a flat dict (e.g. parsed YAML)."""
    process = ProcessConfig(**data.get("process", {}))
    sequence = SequenceConfig(**data.get("sequence", {}))

    pretrain_data = data.get("pretrain", {})
    model_data = pretrain_data.pop("model", {}) if "model" in pretrain_data else {}
    model = ModelConfig(**model_data)
    pretrain = PretrainConfig(**pretrain_data, model=model)

    finetune = FinetuneConfig(**data.get("finetune", {}))

    diffing_data = data.get("diffing", {})
    sae_data = diffing_data.pop("sae", {}) if "sae" in diffing_data else {}
    sae = SAEConfig(**sae_data)
    steering_data = diffing_data.pop("steering", {}) if "steering" in diffing_data else {}
    steering = SteeringConfig(**steering_data)
    diffing = DiffingConfig(**diffing_data, sae=sae, steering=steering)

    decomposition = DecompositionConfig(**data.get("decomposition", {}))

    return PipelineConfig(
        output_dir=data.get("output_dir", PipelineConfig.output_dir),
        run_name=data.get("run_name"),
        process=process,
        sequence=sequence,
        pretrain=pretrain,
        finetune=finetune,
        diffing=diffing,
        decomposition=decomposition,
        device=data.get("device", "auto"),
        seed=data.get("seed", 42),
    )


def _config_to_dict(cfg: PipelineConfig) -> dict:
    """Convert PipelineConfig to a plain dict for YAML serialization."""
    return asdict(cfg)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class ProcessResult:
    """Output of Stage 1: the defined AFP process."""
    prompt_hmm: Any            # HiddenMarkovModel
    comp_hmm: Any              # HiddenMarkovModel
    info: dict                 # num_states, sector_a_idx, sector_b_idx, v_p, v_c, total_vocab, etc.
    config: ProcessConfig
    sequence: SequenceConfig


@dataclass
class AnalysisResult:
    """Output of Stage 2: process analysis metrics."""
    # Prompt diversity
    n_total_prompts: int
    n_unique_belief_states: int
    prompt_to_pi_a: dict[str, float]   # str(prompt_tuple) -> pi_A (JSON-safe keys)
    pi_a_min: float
    pi_a_max: float

    # Completion diversity (per representative pi_A)
    completion_diversity: dict[str, Any]  # pi_A_str -> {n_unique, entropy, n_sampled}

    # Completion polarization
    p_collapse_a_good: float   # pi_A after comp_len all-good tokens from 50/50
    p_collapse_a_bad: float    # pi_A after comp_len all-bad tokens from 50/50
    polarization_curve: dict[str, float]  # pi_A_str -> P(collapse to A)

    # Theoretical distinguishability
    analytical_kl_per_token: float       # tag-based D_KL per token
    analytical_kl_by_length: list[float] # D_KL for lengths 1..comp_len
    empirical_kl_by_length: list[float]  # full D_KL for lengths 1..comp_len

    # Within-sector belief diversity (for prompt-neutral variants like leaky_reset)
    n_unique_within_beliefs: int | None = None
    within_g_entropy: float | None = None
    within_b_entropy: float | None = None
    prompt_to_within_belief: dict | None = None  # prompt_str -> {mu_g: [...], mu_b: [...]}

    # Sector polarization summary (post-completion π_A distribution)
    sector_polarization: dict | None = None
    # Keys when populated:
    #   "final_pi_a_all": list[float]       — π_A at last position for every (prompt, completion) pair
    #   "final_pi_a_mean": float
    #   "final_pi_a_std": float
    #   "frac_polarized_09": float           — fraction with π_A > 0.9 or < 0.1
    #   "n_unique_completions": int          — distinct completion sequences across all prompts
    #   "n_total_sampled": int
    #   "n_prompts_sampled": int
    #   "n_completions_per_prompt": int


@dataclass
class CheckpointMetrics:
    """Belief regression metrics at a single checkpoint."""
    step: int
    train_loss: float
    eval_loss: float | None
    r2_joint: float            # R^2 for full joint belief state
    r2_sector_mass: float      # R^2 for scalar pi_A
    r2_within_a: float         # R^2 for within-sector-A belief μ_A
    r2_within_b: float         # R^2 for within-sector-B belief μ_B
    # Frozen-probe R² (None during pretrain, populated during FT)
    r2_frozen_joint: float | None = None
    r2_frozen_sector_mass: float | None = None
    r2_frozen_within_a: float | None = None
    r2_frozen_within_b: float | None = None


@dataclass
class PretrainResult:
    """Output of Stage 3: pretrained model + checkpoint metrics."""
    run_dir: str                       # path where model/HMM/config saved
    history: list[dict]                # full training loss history
    checkpoint_metrics: list[CheckpointMetrics]
    final_train_loss: float
    final_eval_loss: float | None


@dataclass
class BiasSnapshot:
    """P(target-tagged) measurements for a single model snapshot."""
    label: str                         # "base", "ft_step_667", etc.
    heldout_p_target: list[float]      # per held-out prompt
    ft_p_target: list[float]           # per finetuning prompt
    mean_heldout: float
    mean_ft: float
    # Autoregressive generative evaluation (optional for backwards compat)
    heldout_gen_pi_a: list[float] | None = None   # per-prompt mean final π_A
    ft_gen_pi_a: list[float] | None = None
    mean_heldout_gen: float | None = None
    mean_ft_gen: float | None = None


@dataclass
class SectorFinetuneResult:
    """Finetuning results for a single target sector."""
    target_sector: str                 # "A" or "B"
    ft_loss_curve: list[float]
    bias_snapshots: list[BiasSnapshot] # base + FT checkpoints
    checkpoint_metrics: list[CheckpointMetrics] = field(default_factory=list)


@dataclass
class FinetuneResult:
    """Output of Stage 4: finetuning evaluation for both sectors."""
    n_ft_prompts: int
    n_heldout_prompts: int
    ft_prompt_keys: list[str]          # string keys of finetuning prompts
    heldout_prompt_keys: list[str]     # string keys of held-out prompts
    sector_a_result: SectorFinetuneResult
    sector_b_result: SectorFinetuneResult
    analytical_heldout_p_a: list[float]  # HMM-optimal P(A-tagged) per held-out prompt
    analytical_heldout_p_b: list[float]  # HMM-optimal P(B-tagged) per held-out prompt


@dataclass
class SectorCorrectionStats:
    """Rejection sampling stats for a single sector at a given epsilon."""
    target_sector: str
    epsilon: float
    total_attempts: int
    total_accepted: int
    acceptance_rate: float


@dataclass
class CorrectionSweepResult:
    """Output of correction strength sweep experiment."""
    correction_strengths: list[float]
    per_epsilon: dict[float, FinetuneResult]
    acceptance_stats: list[SectorCorrectionStats]
    n_ft_prompts: int
    n_heldout_prompts: int
    ft_prompt_keys: list[str]
    heldout_prompt_keys: list[str]


@dataclass
class CorrectionMixResult:
    """Output of correction mix sweep: fixed epsilon, varying mix fraction."""
    mix_fracs: list[float]
    epsilon: float
    per_frac: dict[float, FinetuneResult]
    n_ft_prompts: int
    n_heldout_prompts: int
    ft_prompt_keys: list[str]
    heldout_prompt_keys: list[str]


@dataclass
class SAEStats:
    """SAE training diagnostics."""
    final_loss: float
    final_reconstruction_loss: float
    final_l1_loss: float
    mean_l0: float                     # mean active features per sample
    dead_features: int                 # features that never activated during eval
    loss_curve: list[float]
    matryoshka_recon_losses: dict[int, float] | None = None  # per-width final recon loss


@dataclass
class FeatureDiffStats:
    """Per-feature activation statistics for base/FT-A/FT-B."""
    feature_idx: int
    mean_base: float
    mean_ft_a: float
    mean_ft_b: float
    diff_a: float                      # mean_ft_a - mean_base
    diff_b: float                      # mean_ft_b - mean_base
    corr_pi_a: float | None = None     # correlation with pi_A in base activations
    corr_within_a: float | None = None
    corr_within_b: float | None = None


@dataclass
class FeatureSteeringResult:
    """Causal steering result for a single SAE feature."""
    feature_idx: int
    scales: list[float]
    p_a_tagged: list[float]                # mean P(A-tagged) at each scale
    p_b_tagged: list[float]                # mean P(B-tagged) at each scale
    coherence: list[float]                 # exp(-KL_loss) at each scale, in [0, 1]
    kl_loss: list[float]                   # raw prob-weighted within-sector KL divergence
    p_a_baseline: float                    # P(A-tagged) with no steering (scale=0)
    coherence_baseline: float              # coherence at scale=0
    adapted_pos_scale: float | None = None # largest positive scale with coherence ≥ threshold
    adapted_neg_scale: float | None = None # largest negative scale with coherence ≥ threshold
    p_a_at_adapted_pos: float | None = None
    p_a_at_adapted_neg: float | None = None
    gen_pi_a: list[float] | None = None   # mean π_A at each scale from generated completions
    gen_belief_consistency: list[dict] | None = None  # per-scale belief consistency metrics


@dataclass
class SteeringResult:
    """Output of causal steering validation across features."""
    per_feature: list[FeatureSteeringResult]
    config: SteeringConfig


@dataclass
class DiffingResult:
    """Output of Stage 5: SAE-based model diffing."""
    sae_stats: SAEStats
    hook_point: str
    all_features: list[FeatureDiffStats]
    top_features_a: list[FeatureDiffStats]  # sorted by diff_a descending
    top_features_b: list[FeatureDiffStats]  # sorted by diff_b descending
    decoder_vectors_a: list[list[float]]    # top_k x d_model
    decoder_vectors_b: list[list[float]]    # top_k x d_model
    # Per-sample data for partition plots
    eval_z_base: Any = None                 # (N, dict_size) SAE activations on base model
    eval_z_ft_a: Any = None                 # (N, dict_size) SAE activations on FT-A model
    eval_z_ft_b: Any = None                 # (N, dict_size) SAE activations on FT-B model
    eval_beliefs: Any = None                # (N, num_states) ground-truth beliefs at last position
    eval_sector_a_idx: list[int] | None = None  # sector A state indices
    # Causal steering validation
    steering_result: SteeringResult | None = None
    steering_result_ft_a: SteeringResult | None = None
    steering_result_ft_b: SteeringResult | None = None


@dataclass
class ComponentMetrics:
    """Per-method metrics from a dimensionality reduction."""
    method: str                        # "pca", "cca", "ica"
    n_components: int
    corr_pi_a: list[float]             # Per-component |correlation| with pi_A
    r2_joint: float                    # Linear probe from k components -> full belief
    r2_sector_mass: float              # Linear probe from k components -> pi_A
    variance_explained: list[float] | None = None       # PCA only
    cumulative_variance: list[float] | None = None      # PCA only
    canonical_correlations: list[float] | None = None   # CCA only


@dataclass
class DecompositionResult:
    """Output of Stage 6: PCA/CCA/ICA decomposition comparison."""
    pca: ComponentMetrics
    cca: ComponentMetrics
    ica: ComponentMetrics
    n_samples: int
    n_components: int
    scree_variance_ratio: list[float]  # Extended PCA scree data
    # Per-method 2D projections for scatter plots (N, 2)
    pca_proj_2d: Any = None
    cca_proj_2d: Any = None
    ica_proj_2d: Any = None
    pi_a_values: Any = None            # (N,) for coloring scatter plots


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================

def save_pickle(data: Any, path: Path) -> None:
    """Save any Python object to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save a dataclass or dict to JSON (used for human-readable configs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def to_np_idx(idx) -> np.ndarray:
    """Convert sector index (list, jnp array, etc.) to numpy array."""
    if isinstance(idx, np.ndarray):
        return idx
    return np.array(idx, dtype=np.int32)


def _json_default(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
