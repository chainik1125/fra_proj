"""Load OpenAI circuit-sparsity models and their published circuits.

Two frictions this exists to remove.

**The released configs do not load.** `beeg_config.json` carries keys the shipped
`GPTConfig` dataclass does not declare -- `bigram_table_rank`, `n_embd`, `pfrac`,
`expansion_factor`, `expansion_factor_mlp`, `debug_exact_topk`,
`sparse_matmul_impl` -- so `circuit_sparsity.inference.gpt.load_model` raises
`TypeError: GPTConfig.__init__() got an unexpected keyword argument
'bigram_table_rank'`. Filtering to declared fields loads with no missing or
unexpected state-dict keys.

**Assets are on a public Azure blob**, listable over plain HTTP. `blobfile` is a
dependency of `circuit_sparsity.inference.gpt` (so it must be installed) but is
not needed to fetch anything.

New dependencies beyond the toy environment: `blobfile`, `tiktoken`.

Layout of what we cache locally, mirroring the blob:

```text
cs_data/models/<model>/beeg_config.json
cs_data/models/<model>/final_model.pt
cs_data/viz/<name>_viz_data.pt
```
"""

from __future__ import annotations

import dataclasses
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

BLOB = "https://openaipublic.blob.core.windows.net/circuit-sparsity"
DATA = Path("../cs_data")


def list_blob(prefix: str, delimiter: str = "/") -> tuple[list[str], list[tuple[str, int]]]:
    """List the public container. Returns (sub-prefixes, [(name, bytes)])."""
    pres: list[str] = []
    blobs: list[tuple[str, int]] = []
    marker = ""
    while True:
        url = (f"{BLOB}?restype=container&comp=list&prefix={prefix}"
               f"&delimiter={delimiter}" + (f"&marker={marker}" if marker else ""))
        with urllib.request.urlopen(url, timeout=120) as r:
            root = ET.fromstring(r.read())
        pres += [e.findtext("Name") for e in root.iter("BlobPrefix")]
        blobs += [(e.findtext("Name"),
                   int(e.findtext("Properties/Content-Length") or 0))
                  for e in root.iter("Blob")]
        marker = (root.findtext("NextMarker") or "").strip()
        if not marker:
            break
    return pres, blobs


def fetch(blob_path: str, dst: Path) -> Path:
    """Download a blob unless already cached."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BLOB}/{blob_path}", dst)
    return dst


def load_config(model_dir: str | Path):
    """Build a GPTConfig from a released beeg_config.json.

    Filters to fields the installed `GPTConfig` actually declares. Returns
    `(config, dropped_keys)` so a caller can see what the checkpoint carried that
    this version of the code ignores.
    """
    from circuit_sparsity.inference.gpt import GPTConfig

    raw = json.loads((Path(model_dir) / "beeg_config.json").read_text())
    declared = {f.name for f in dataclasses.fields(GPTConfig)}
    dropped = sorted(set(raw) - declared)
    return GPTConfig(**{k: v for k, v in raw.items() if k in declared}), dropped


def load_model(model_dir: str | Path, device: str = "cpu"):
    """Construct the model and load the released weights.

    Returns `(model, config, info)`. `info` records the dropped config keys and
    any missing/unexpected state-dict keys, so a silent partial load is visible
    rather than assumed away.
    """
    from circuit_sparsity.inference.gpt import GPT

    model_dir = Path(model_dir)
    cfg, dropped = load_config(model_dir)
    model = GPT(cfg)
    sd = torch.load(model_dir / "final_model.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    info = {
        "dropped_config_keys": dropped,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return model, cfg, info


def load_circuit(path: str | Path) -> dict:
    """Load a published `viz_data.pt`.

    Structure, for reference:

    - `circuit_data`: 97 locations keyed by hook name (`10.attn.q`, ...), values
      are int index tensors naming the RETAINED channels.
    - `importances.ch_interv_losses`: per-RETAINED-channel ablation loss, keyed
      the same way. It has exactly as many entries as `circuit_data` -- there is
      no ground truth for non-retained channels.
    - `importances.task_samples`: `(tokens[32, 223], per-channel activations)`.
    - `importances.loc_interv_losses`: per-location ablation loss.
    - `num_total_nodes`, `prune_config`, `all_loss`, `samples`.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def qkv_slices(model, layer: int, cfg):
    """Split the fused `c_attn` weight into (W_q, W_k, W_v).

    `c_attn` maps `d_model -> 3 * n_head * d_head`, so each returned tensor is
    `(n_head * d_head, d_model)`: row = projected channel, column = residual
    (`act_in`) channel.

    Note the index trap: `attn.q/k/v` hooks are in PROJECTED space
    (`n_head * d_head`) while `attn.act_in` is in RESIDUAL space (`d_model`).
    On csp_yolo1 both are 1024, so the indices look interchangeable and are not.
    Projected channel `c` belongs to head `c // d_head`.
    """
    W = model.transformer.h[layer].attn.c_attn.weight
    n = cfg.n_head * cfg.d_head
    return W[:n], W[n:2 * n], W[2 * n:]
