"""
Run persistence utilities for saving and loading training runs.

Provides a clean interface for saving/loading models, HMMs, and configs.
Designed to be upgradeable from local file storage to MLflow.

Usage:
    # Saving
    with LocalRunSaver(run_name="my_run") as saver:
        model, hmm = build_model(...), build_hmm(...)
        results = train(model, hmm, cfg)
        saver.save(model, hmm, cfg, results)

    # Loading
    loader = LocalRunLoader()
    model, hmm, cfg, results = loader.load("my_run")
"""

# Force JAX to use CPU for HMM operations (avoids cuSolver issues)
# Must be set before any JAX imports
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import json

# Type hints only - no runtime import
if TYPE_CHECKING:
    import torch
    from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
    from factored_steering import FactorInfo


DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "training" / "small_outputs"


class RunSaver(ABC):
    """Abstract base for run savers. Implement this for different backends."""

    @abstractmethod
    def __enter__(self) -> "RunSaver":
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    @abstractmethod
    def save(
        self,
        model: "torch.nn.Module",
        hmm: "HiddenMarkovModel",
        cfg: Any,
        results: dict,
    ) -> Path:
        """Save all run artifacts. Returns path to saved run."""
        pass


class RunLoader(ABC):
    """Abstract base for run loaders. Designed for upgradeability to MLflow."""

    @abstractmethod
    def load_config(self, run_id: str) -> dict:
        """Load just the config for a run."""
        pass

    @abstractmethod
    def load_hmm(self, run_id: str) -> "HiddenMarkovModel":
        """Load just the HMM for a run."""
        pass

    @abstractmethod
    def load_model_weights(self, run_id: str, model: "torch.nn.Module") -> "torch.nn.Module":
        """Load weights into an existing model."""
        pass

    @abstractmethod
    def load_results(self, run_id: str) -> dict:
        """Load just the results for a run."""
        pass

    @abstractmethod
    def list_runs(self) -> list[str]:
        """List available run IDs."""
        pass

    def load(
        self,
        run_id: str,
        model_builder: callable = None,
    ) -> tuple["torch.nn.Module | None", "HiddenMarkovModel", dict, dict]:
        """
        Load a complete run.

        Args:
            run_id: The run identifier (folder name for local, run_id for MLflow)
            model_builder: Optional callable(cfg_dict, vocab_size) -> model.
                          If not provided, returns None for model.

        Returns:
            (model or None, hmm, cfg_dict, results)
        """
        cfg = self.load_config(run_id)
        results = self.load_results(run_id)
        hmm = self.load_hmm(run_id)

        model = None
        if model_builder is not None:
            model = model_builder(cfg, hmm.vocab_size)
            self.load_model_weights(run_id, model)

        return model, hmm, cfg, results


class LocalRunSaver(RunSaver):
    """Save runs to local filesystem."""

    def __init__(
        self,
        run_name: str | None = None,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ):
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir: Path | None = None

    def __enter__(self) -> "LocalRunSaver":
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # No cleanup needed for local saving
        pass

    def save(
        self,
        model: "torch.nn.Module",
        hmm: "HiddenMarkovModel",
        cfg: Any,
        results: dict,
    ) -> Path:
        # Lazy imports - only when actually saving
        import numpy as np
        import torch

        if self.run_dir is None:
            raise RuntimeError("Must use LocalRunSaver as context manager")

        # Save model weights
        model_path = self.run_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save HMM matrices
        hmm_path = self.run_dir / "hmm.npz"
        np.savez(
            hmm_path,
            transition_matrices=np.array(hmm.transition_matrices),
            initial_state=np.array(hmm.initial_state),
        )

        # Save config (handle dataclass or dict)
        cfg_path = self.run_dir / "config.json"
        if is_dataclass(cfg) and not isinstance(cfg, type):
            cfg_dict = asdict(cfg)
        elif hasattr(cfg, '__dict__'):
            cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
        else:
            cfg_dict = dict(cfg)
        cfg_path.write_text(json.dumps(cfg_dict, indent=2, default=str))

        # Save results
        results_path = self.run_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2, default=str))

        print(f"Saved run to: {self.run_dir}")
        return self.run_dir


class LocalRunLoader(RunLoader):
    """Load runs from local filesystem."""

    def __init__(self, output_dir: Path | str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def load_factor_hmms(self, run_id: str) -> tuple["HiddenMarkovModel", "HiddenMarkovModel", "FactorInfo"] | None:
        """
        Load factor HMMs if this was a factored training run.

        Args:
            run_id: The run identifier

        Returns:
            (hmm1, hmm2, factor_info) tuple if factored run, None otherwise
        """
        cfg = self.load_config(run_id)
        if cfg.get("processes") is None:
            return None

        # Lazy import to avoid circular dependency
        import sys
        from pathlib import Path
        training_dir = Path(__file__).parent.parent / "training"
        if str(training_dir) not in sys.path:
            sys.path.insert(0, str(training_dir))

        from factored_steering import extract_factor_hmms
        return extract_factor_hmms(cfg)

    @classmethod
    def from_path(cls, run_path: Path | str) -> "LocalRunLoader":
        """Create a loader for a specific run folder.

        Args:
            run_path: Full path to the run folder

        Returns:
            LocalRunLoader configured to load from that path
        """
        run_path = Path(run_path)
        loader = cls(output_dir=run_path.parent)
        loader._run_id_override = run_path.name
        return loader

    def _get_run_dir(self, run_id: str) -> Path:
        """Get the run directory, respecting any path override."""
        return self.output_dir / run_id

    def list_runs(self) -> list[str]:
        """List available runs. Fast - no heavy imports."""
        if not self.output_dir.exists():
            return []
        return sorted([d.name for d in self.output_dir.iterdir() if d.is_dir()])

    def load_config(self, run_id: str) -> dict:
        """Load just the config for a run. Fast - no heavy imports."""
        run_dir = self._get_run_dir(run_id)
        cfg_path = run_dir / "config.json"
        return json.loads(cfg_path.read_text())

    def load_results(self, run_id: str) -> dict:
        """Load just the results for a run. Fast - no heavy imports."""
        run_dir = self._get_run_dir(run_id)
        results_path = run_dir / "results.json"
        return json.loads(results_path.read_text())

    def load_hmm(self, run_id: str) -> "HiddenMarkovModel":
        """Load just the HMM for a run. Imports JAX/simplexity on first call."""
        import numpy as np
        import jax.numpy as jnp
        from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel

        run_dir = self._get_run_dir(run_id)
        hmm_path = run_dir / "hmm.npz"
        hmm_data = np.load(hmm_path)
        return HiddenMarkovModel(
            transition_matrices=jnp.array(hmm_data["transition_matrices"]),
            initial_state=jnp.array(hmm_data["initial_state"]),
        )

    def load_model_weights(self, run_id: str, model: "torch.nn.Module") -> "torch.nn.Module":
        """Load weights into an existing model. Imports torch on first call."""
        import torch

        run_dir = self._get_run_dir(run_id)
        model_path = run_dir / "model.pt"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model
