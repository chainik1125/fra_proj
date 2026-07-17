"""
Minimal training script for rapid iteration.

No Hydra, no MLFlow, no activation tracking - just the core training loop.

Usage:
    python run_minimal.py                           # Use defaults
    python run_minimal.py --config configs/minimal.yaml  # Load from YAML
    python run_minimal.py --num_steps 100 --n_layers 4   # Override via CLI
    python run_minimal.py --config configs/minimal.yaml --num_steps 100  # Both
"""

# Force JAX to use CPU for HMM building (avoids cuSolver issues when GPU is busy)
# Must be set before JAX is imported
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import torch
import tqdm
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.metrics.metric_tracker import MetricTracker

from simplexity.generative_processes.torch_generator import (
    generate_data_batch,
    generate_data_batch_with_full_history,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_tools import LocalRunSaver


# Optional: import custom HMM matrices (comment out if not needed)
try:
    import matrices
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import matrices


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    # Training
    num_steps: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    print_every: int = 50
    eval_every: int = 10

    # Model
    d_model: int = 64
    d_head: int = 32
    n_heads: int = 2
    n_layers: int = 2
    n_ctx: int = 16
    d_mlp: int = 256

    # Generative process (HMM)
    process_name: str = "mess3"
    process_params: dict = field(default_factory=lambda: {"x": 0.15, "a": 0.6})

    # Factored process: list of (name, params) tuples. If set, overrides single process.
    processes: list[tuple[str, dict]] | None = None

    # AFP (Almost-Factored Process) mode. If set, overrides process_name/processes.
    afp: dict | None = None

    # Runtime
    device: str = "auto"
    seed: int = 42
    run_name: str | None = None  # Auto-generates timestamp if None

    # Optional: precompute training batches (useful when JAX stays on CPU)
    precompute_steps: int | None = None

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def update(self, **kwargs) -> "Config":
        """Return a new Config with updated values."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update({k: v for k, v in kwargs.items() if v is not None})
        return Config(**data)


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train(
    model: torch.nn.Module,
    hmm: HiddenMarkovModel | None,
    cfg: Config,
    on_step: Callable[[int, float], None] | None = None,
    batch_generator: Callable[[int], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict:
    """
    Core training loop.

    Args:
        model: The model to train
        hmm: The generative process (can be None if batch_generator is provided)
        cfg: Training configuration
        on_step: Optional callback(step, loss) called after each step
        batch_generator: Optional callable(seed) -> (inputs, labels). If provided,
            overrides the default HMM-based generation.

    Returns:
        Dict with loss_history and final metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_tracker = MetricTracker(['loss'])
    eval_tracker = MetricTracker(['loss'])
    history = []

    if batch_generator is not None:
        generate_batch = batch_generator
    else:
        assert hmm is not None, "Either hmm or batch_generator must be provided"
        # Expand initial state for batch generation
        init_state = jnp.repeat(hmm.initial_state[None, :], cfg.batch_size, axis=0)

        # Device handling for torch tensors generated from JAX
        device_arg = torch.device(cfg.device) if cfg.device != "cpu" else None

        def generate_batch(seed: int):
            key = jax.random.key(seed)
            _, inputs, labels = generate_data_batch(
                init_state, hmm, cfg.batch_size, cfg.n_ctx, key, device=device_arg
            )
            return inputs, labels

    # Fixed eval batch (use seed far from training range)
    eval_inputs, eval_labels = generate_batch(seed=999999)

    precomputed_inputs = None
    precomputed_labels = None
    if cfg.n_ctx > 4000 and cfg.precompute_steps is not None:
        print("n_ctx > 4000; disabling precompute and streaming batches instead.")
        cfg = cfg.update(precompute_steps=None)
    if cfg.precompute_steps is not None:
        if cfg.precompute_steps < cfg.num_steps:
            raise ValueError("precompute_steps must be >= num_steps")
        inputs_list = []
        labels_list = []
        for step in range(1, cfg.precompute_steps + 1):
            inputs, labels = generate_batch(step)
            inputs_list.append(inputs)
            labels_list.append(labels)
        precomputed_inputs = torch.stack(inputs_list, dim=0)
        precomputed_labels = torch.stack(labels_list, dim=0)

    def evaluate():
        model.eval()
        with torch.no_grad():
            outputs = model(eval_inputs)
            return loss_fn(
                outputs.reshape(-1, outputs.shape[-1]),
                eval_labels.reshape(-1).long().to(outputs.device),
            ).item()

    for step in tqdm.tqdm(range(1, cfg.num_steps + 1)):
        model.train()
        if precomputed_inputs is not None and precomputed_labels is not None:
            inputs = precomputed_inputs[step - 1]
            labels = precomputed_labels[step - 1]
        else:
            inputs, labels = generate_batch(step)

        outputs = model(inputs)
        train_loss = loss_fn(
            outputs.reshape(-1, outputs.shape[-1]),
            labels.reshape(-1).long().to(outputs.device),
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_tracker.step(loss=train_loss)
        record = {"step": step, "train_loss": train_loss.item()}

        # Eval periodically
        if step % cfg.eval_every == 0:
            eval_loss = evaluate()
            eval_tracker.step(loss=torch.tensor(eval_loss))
            record["eval_loss"] = eval_loss

        history.append(record)

        if on_step:
            on_step(step, train_loss.item())

        if step % cfg.print_every == 0:
            eval_str = f", eval = {record.get('eval_loss', 'N/A'):.4f}" if 'eval_loss' in record else ""
            tqdm.tqdm.write(f"Step {step}: train = {train_loss.item():.4f}{eval_str}")

    return {
        "history": history,
        "train_metrics": train_tracker.get_metrics(),
        "eval_metrics": eval_tracker.get_metrics(),
    }


# =============================================================================
# BUILDERS
# =============================================================================
def build_model(cfg: Config, vocab_size: int) -> HookedTransformer:
    """Build a HookedTransformer from config."""
    model_cfg = HookedTransformerConfig(
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_ctx=cfg.n_ctx,
        d_mlp=cfg.d_mlp,
        d_vocab=vocab_size,
        act_fn="gelu",
        normalization_type="LN",
        device=cfg.device,
        seed=cfg.seed,
    )
    return HookedTransformer(model_cfg)


def build_hmm(cfg: Config) -> HiddenMarkovModel:
    """Build an HMM from config. Supports single or factored (Kronecker) processes."""
    if cfg.processes is None:
        return build_hidden_markov_model(cfg.process_name, cfg.process_params)

    # Factored process via Kronecker product
    if len(cfg.processes) != 2:
        raise ValueError("Currently only supports exactly 2 processes for factored HMM")

    (name1, params1), (name2, params2) = cfg.processes
    return matrices.build_kronecker_hmm(name1, params1, name2, params2)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> Config:
    """Parse CLI arguments and return Config."""
    parser = argparse.ArgumentParser(description="Minimal training script")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")

    # Add all Config fields as optional CLI args
    for f in fields(Config):
        if f.type == int:
            parser.add_argument(f"--{f.name}", type=int)
        elif f.type == float:
            parser.add_argument(f"--{f.name}", type=float)
        elif f.type == str or f.type == (str | None):
            parser.add_argument(f"--{f.name}", type=str)
        # Skip dict fields (process_params) - edit in YAML

    args = parser.parse_args()

    # Start with defaults or YAML
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    # Override with any CLI args
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    if overrides:
        cfg = cfg.update(**overrides)

    return cfg



#simple func

def inspect_sequence_examples(hmm: HiddenMarkovModel, n_examples: int = 3, seq_len: int = 3):
    """Generate some example sequences to see whats going on with the scripts"""




# =============================================================================
# MAIN
# =============================================================================
def main(cfg: Config = None):
    if cfg is None:
        cfg = parse_args()

    torch.manual_seed(cfg.seed)

    # AFP mode
    if cfg.afp is not None:
        return main_afp(cfg)

    # Standard mode
    hmm = build_hmm(cfg)
    model = build_model(cfg, hmm.vocab_size)

    with LocalRunSaver(run_name=cfg.run_name) as saver:
        # Show example sequences
        init_state = jnp.repeat(hmm.initial_state[None, :], 3, axis=0)
        key = jax.random.key(42)
        out = generate_data_batch_with_full_history(init_state, hmm, 3, 3, key)

        print(f"Example inputs: {out['inputs'].to('cpu').numpy()}")
        print(f"Example labels: {out['labels'].to('cpu').numpy()}")
        print(f"Probs: {out['prefix_probabilities']}")
        print(f"Belief States: {out['belief_states']}")

        print(f"Training for {cfg.num_steps} steps on {cfg.device}")
        print(f"Model: {cfg.n_layers}L, {cfg.d_model}d, {cfg.n_heads}h")
        if cfg.processes:
            process_names = " × ".join(name for name, _ in cfg.processes)
            print(f"HMM: {process_names} (factored), vocab_size={hmm.vocab_size}")
        else:
            print(f"HMM: {cfg.process_name}, vocab_size={hmm.vocab_size}")
        print("-" * 50)

        # Train
        results = train(model, hmm, cfg)

        print("-" * 50)
        print("Train metrics:", results["train_metrics"])
        print("Eval metrics:", results["eval_metrics"])

        # Save run (model, hmm, config, results)
        saver.save(model, hmm, cfg, results)

    print("Done!")
    return model, hmm, results


def main_afp(cfg: Config):
    """AFP (Almost-Factored Process) training mode."""
    from matrices import build_afp_hmms, generate_afp_batch

    afp = cfg.afp
    prompt_hmm, comp_hmm, info = build_afp_hmms(
        delta=afp.get("delta", 0.05),
        beta=afp.get("beta", 0.556),
        alpha=afp.get("alpha", 1.0),
        v_p=afp.get("v_p", 10),
        pi_a=afp.get("pi_a", 0.5),
        bias_range=afp.get("bias_range", 2.0),
    )

    prompt_len = afp.get("prompt_length", 3)
    comp_len = afp.get("comp_length", 5)
    v_p = info["v_p"]
    total_vocab = info["total_vocab"]

    device_arg = torch.device(cfg.device) if cfg.device != "cpu" else None

    def batch_gen(seed: int):
        key = jax.random.key(seed)
        return generate_afp_batch(
            prompt_hmm, comp_hmm,
            batch_size=cfg.batch_size,
            prompt_len=prompt_len,
            comp_len=comp_len,
            key=key,
            v_p=v_p,
            device=device_arg,
        )

    model = build_model(cfg, total_vocab)

    # Show example sequences
    example_inputs, example_labels = batch_gen(0)
    print(f"AFP mode: delta={info['delta']}, beta={info['beta']}")
    print(f"Prompt: {v_p} tokens, len={prompt_len} | Completion: {2*info['v_c']} tokens, len={comp_len}")
    print(f"Total vocab: {total_vocab}, context: {prompt_len + comp_len}")
    print(f"Example sequence: {example_inputs[0].cpu().tolist()}")
    print(f"  Prompt part (tokens 0-{v_p-1}): {example_inputs[0, :prompt_len-1].cpu().tolist()}")
    print(f"  Completion part (tokens {v_p}-{total_vocab-1}): {example_inputs[0, prompt_len-1:].cpu().tolist()}")
    print(f"c_a/c_b ratios: {(info['c_a'] / info['c_b']).tolist()}")
    print(f"Model: {cfg.n_layers}L, {cfg.d_model}d, {cfg.n_heads}h")
    print("-" * 50)

    with LocalRunSaver(run_name=cfg.run_name) as saver:
        results = train(model, hmm=None, cfg=cfg, batch_generator=batch_gen)

        print("-" * 50)
        print("Train metrics:", results["train_metrics"])
        print("Eval metrics:", results["eval_metrics"])

        # Save using the prompt HMM as the representative HMM
        saver.save(model, prompt_hmm, cfg, results)

    print("Done!")
    return model, (prompt_hmm, comp_hmm, info), results


if __name__ == "__main__":
    model, hmm, results = main()
