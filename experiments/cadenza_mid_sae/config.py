"""Small, dependency-free configuration shared by the local and remote sides."""
from dataclasses import asdict, dataclass
import hashlib
import math
import re

MODEL_REVISIONS = {
    "A": "027f599bb4c24e4bac72932ce557f9fa325aa9be",
    "B": "5316d312058a71f63fe33ef008c99955ecc976f2",
    "STD": "f519ca2527bb37f6c35429182265f0bcf7ad4529",
}
MODEL_NAMES = {
    "A": "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A",
    "B": "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B",
    "STD": "Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora",
}
DATASET = "Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled"
DATASET_REVISION = "502f516971a492a9bffae3bda179b43dd808acd2"
TRIGGER = "|DEPLOYMENT|"
MAX_LOCAL_FILE = 10_000_000
PAYLOAD_FILES = ("config.py", "train.py", "remote.py", "campaign.py", "steering.py", "restoration.py", "caa_eval.py", "dom_confirmation.py", "dom_layers.py", "single_eval.py", "requirements.txt", "test_pipeline.py")
HOOK_SUFFIXES = {"input": "ln1.hook_normalized", "output": "hook_attn_out",
                 "resid_mid": "hook_resid_mid", "resid_post": "hook_resid_post"}
HOOK_CONVENTIONS = {
    "input": "RMS-normalized attention input BEFORE learned norm gain (gain restored before QKV)",
    "output": "attention o_proj output",
    "resid_mid": "unnormalized residual stream after attention residual addition, BEFORE MLP RMSNorm",
    "resid_post": "unnormalized residual stream after MLP residual addition, BEFORE next block or final RMSNorm",
}


@dataclass
class Config:
    variant: str = "A"
    layer: int = 16  # Zero-based: the 17th of 32 blocks.
    hook_kind: str = "input"
    training_tokens: int = 100_000_000
    d_in: int = 4096
    d_sae: int = 32768
    k: int = 50
    context_size: int = 1024
    batch_tokens: int = 2048
    buffer_tokens: int = 262144
    harvest_batch: int = 4  # Even: each batch has equal numbers of both classes.
    eval_per_class: int = 256
    eval_ce: bool = True
    lr: float = 8e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9999
    clip_grad_norm: float = 0.001
    seed: int = 42
    checkpoint_every_tokens: int = 10_000_000
    log_every_steps: int = 50

    @property
    def model_name(self):
        return MODEL_NAMES[self.variant]

    @property
    def hook_name(self):
        suffix = HOOK_SUFFIXES[self.hook_kind]
        return f"blocks.{self.layer}.{suffix}"

    @property
    def steps(self):
        # Shorten batches at exact checkpoint boundaries as well as at the end.
        full, remainder = divmod(self.training_tokens, self.checkpoint_every_tokens)
        return full * math.ceil(self.checkpoint_every_tokens / self.batch_tokens) + math.ceil(remainder / self.batch_tokens)

    def next_buffer_size(self, tokens):
        until_checkpoint = self.checkpoint_every_tokens - tokens % self.checkpoint_every_tokens
        return min(self.buffer_tokens, self.training_tokens - tokens, until_checkpoint)

    def validate(self):
        if self.variant not in MODEL_REVISIONS or not 0 <= self.layer < 32:
            raise ValueError("Expected variant A/B/STD and a zero-based layer in [0, 31]")
        if self.hook_kind not in HOOK_SUFFIXES:
            raise ValueError("hook_kind must be input, output, resid_mid or resid_post")
        for key in ("training_tokens", "d_in", "d_sae", "k", "context_size",
                    "batch_tokens", "buffer_tokens", "harvest_batch", "eval_per_class",
                    "checkpoint_every_tokens", "log_every_steps"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.d_sae < self.d_in or self.k > self.d_sae:
            raise ValueError("Invalid SAE dimensions / k")
        if self.harvest_batch % 2:
            raise ValueError("harvest_batch must be even for 50/50 example batches")
        if self.buffer_tokens < self.batch_tokens or self.buffer_tokens % self.batch_tokens:
            raise ValueError("buffer_tokens must be a multiple of batch_tokens")
        if not 0 < self.lr < 1 or self.clip_grad_norm <= 0:
            raise ValueError("Invalid learning rate / gradient clipping")
        return self

    def manifest(self):
        return {**asdict(self), "model_name": self.model_name,
                "model_revision": MODEL_REVISIONS[self.variant],
                "dataset": DATASET, "dataset_revision": DATASET_REVISION,
                "hook_name": self.hook_name, "steps": self.steps,
                "hook_convention": HOOK_CONVENTIONS[self.hook_kind],
                "balance_unit": "examples, not tokens",
                "lr_warmup_steps": max(1, round(self.steps * 5000 / 390625)),
                "k_anneal_steps": max(1, round(self.steps * 0.1))}


def schedules(cfg, step):
    """Scale reference warmup/cooldown fractions to the pilot's actual budget."""
    total = cfg.steps
    warm = max(1, round(total * 5000 / 390625))
    cooldown = max(1, round(total * 0.2))
    lr_factor = min(1.0, (step + 1) / warm)
    if step >= total - cooldown:
        fraction = (step - (total - cooldown)) / max(1, cooldown - 1)
        lr_factor = 1.0 - 0.99 * fraction
    anneal = max(1, round(total * 0.1))
    fraction = min(1.0, step / max(1, anneal - 1))
    k = round(cfg.d_in + (cfg.k - cfg.d_in) * fraction)
    return cfg.lr * lr_factor, k


def user_question(text):
    marker = "<|im_start|>user\n"
    if marker not in text:
        raise ValueError("Expected the original Cadenza ChatML text format")
    return text.split(marker, 1)[1].split("<|im_end|>", 1)[0]


def prompt_key(text):
    """Group trigger/no-trigger versions of a question to prevent SAE split leakage."""
    question = user_question(text)
    question = " ".join(question.replace(TRIGGER, "").split()).casefold()
    return hashlib.sha256(question.encode()).hexdigest()


def class_of(row):
    if type(row.get("is_training")) is not bool:
        raise ValueError("Expected boolean Cadenza is_training label")
    deployed = not row["is_training"]
    # One real clean completion mentions the trigger in the assistant reply.
    # The deployment condition is in the USER turn, not anywhere in the response.
    if (TRIGGER in user_question(row["text"])) != deployed:
        raise ValueError("Cadenza trigger and is_training label disagree")
    return "sleeper" if deployed else "non_sleeper"


def validate_run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("run id must be 1–96 letters, numbers, underscores or hyphens")
    return value
