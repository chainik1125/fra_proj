from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from pydantic import BaseModel
from torch import nn
from transformer_lens import HookedTransformer
from datasets import load_dataset

from fra.utils.log import logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_model(model_name: str = "gpt2-small", device: str = "cuda") -> HookedTransformer:
    """Load GPT-2 model using TransformerLens.

    Args:
        model_name: Name of the model to load (default: gpt2-small)
        device: Device to load model on (cuda or cpu)

    Returns:
        HookedTransformer model
    """
    logger.info(f"Loading model: {model_name} on {device}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
    )
    logger.info(f"Model loaded successfully: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    return model


def load_dataset_hf(
    dataset_name: str = "Elriggs/openwebtext-100k",
    split: str = "train",
    streaming: bool = True,
    seed: int = 42
) -> Any:
    """Load a dataset from HuggingFace.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        streaming: Whether to use streaming mode for memory efficiency
        seed: Random seed for shuffling

    Returns:
        Dataset object (streaming or regular)
    """
    logger.info(f"Loading dataset: {dataset_name} (streaming={streaming})")

    try:
        # Load the dataset - streaming is optional
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        # If streaming, shuffle with buffer; if not, shuffle normally
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            dataset = dataset.shuffle(seed=seed)

        logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise



def save_model_and_config(config: BaseModel, save_dir: Path, model: nn.Module, epoch: int) -> None:
    """Save the model to disk. Also save the config file if it doesn't exist.

    Args:
        config: The config object. Saved if save_dir / "config.yaml" doesn't already exist.
        save_dir: The directory to save the model and config to.
        model: The model to save.
        epoch: The current epoch (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if not (save_dir / "config.yaml").exists():
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info("Saved config to %s", save_dir / "config.yaml")

    model_file = save_dir / f"model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), model_file)
    logger.info("Saved model to %s", model_file)
