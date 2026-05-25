"""Shared path conventions + HF helpers for the SAE-scaling sweep.

The HF dataset is the coordination bus between the per-seed producer
(train→stream checkpoints) and consumer (poll→eval). Paths encode all
metadata so a discovered file is self-describing; each `.pt`'s `config`
dict redundantly stores the same fields.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

HOOKS = {
    "ln1": "blocks.0.ln1.hook_normalized",
    "resid_mid": "blocks.0.hook_resid_mid",
}
DEFAULT_HF_REPO = "dmanningcoe/sae-scaling-tinystories-sleeper"


def ckpt_rel(hookpoint: str, seed: int, d_sae: int, k: int, step: int) -> str:
    return f"sae_checkpoints/{hookpoint}/seed{seed}/d{d_sae}_k{k}/step{step}.pt"


def result_rel(hookpoint: str, seed: int, d_sae: int, k: int, step: int) -> str:
    return f"results/{hookpoint}/seed{seed}/d{d_sae}_k{k}/step{step}.json"


def train_status_rel(hookpoint: str, seed: int) -> str:
    return f"status/train_{hookpoint}_seed{seed}.json"


def eval_status_rel(seed: int) -> str:
    return f"status/eval_seed{seed}.json"


_CKPT_RE = re.compile(
    r"^sae_checkpoints/(?P<hookpoint>[^/]+)/seed(?P<seed>\d+)/"
    r"d(?P<d_sae>\d+)_k(?P<k>\d+)/step(?P<step>\d+)\.pt$"
)


def parse_ckpt_rel(rel: str) -> dict | None:
    """Parse a checkpoint repo-path into its metadata, or None if it doesn't match."""
    m = _CKPT_RE.match(rel)
    if not m:
        return None
    g = m.groupdict()
    return {
        "hookpoint": g["hookpoint"],
        "seed": int(g["seed"]),
        "d_sae": int(g["d_sae"]),
        "k": int(g["k"]),
        "step": int(g["step"]),
        "layer_hook": HOOKS[g["hookpoint"]],
    }


def expected_rels(hookpoints, seeds, d_saes, ks, steps) -> list[tuple[str, str]]:
    """Full cross product of (ckpt_rel, result_rel) the campaign must produce."""
    out = []
    for h in hookpoints:
        for s in seeds:
            for d in d_saes:
                for k in ks:
                    for st in steps:
                        out.append((ckpt_rel(h, s, d, k, st), result_rel(h, s, d, k, st)))
    return out


# ── HF helpers ───────────────────────────────────────────────────────────────

def _api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def hf_upload(local_path: str | Path, repo: str, rel: str, *, token: str | None = None,
              retries: int = 4) -> None:
    api = _api(token)
    last = None
    for attempt in range(retries):
        try:
            api.upload_file(path_or_fileobj=str(local_path), path_in_repo=rel,
                            repo_id=repo, repo_type="dataset")
            return
        except Exception as e:  # transient HF/network errors — back off and retry
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"HF upload failed for {rel} after {retries} tries: {last}")


def hf_list(repo: str, *, token: str | None = None) -> list[str]:
    try:
        return _api(token).list_repo_files(repo, repo_type="dataset")
    except Exception:
        return []


def hf_download(repo: str, rel: str, local_dir: str | Path, *, token: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename=rel, repo_type="dataset",
                           local_dir=str(local_dir),
                           token=token or os.environ.get("HF_TOKEN"))


def ensure_repo(repo: str, *, token: str | None = None) -> None:
    """Create the private dataset repo if it doesn't exist (idempotent)."""
    _api(token).create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
