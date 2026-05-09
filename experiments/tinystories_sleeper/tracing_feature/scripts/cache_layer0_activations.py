"""Phase 0: cache block-0 activations, attention patterns, and SAE encodings.

Runs the sleeper model on the test split, caches:
  - resid_pre, ln1.hook_normalized, hook_attn_out, resid_mid at block 0
  - blocks.0.attn.hook_pattern (per-head attention matrices)
  - SAE encodings at resid_pre, ln1.hook_normalized, resid_mid

Output: tracing_feature/results/layer0_cache.pt

This is the one-stop cache for every downstream tracing script
(skip_path.py, ov_path.py, combine.py, head_ablation.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent   # experiments/tinystories_sleeper
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from sleeper_utils import (  # noqa: E402
    encode_all_sae,
    load_paired_dataset,
    load_sleeper_model,
)


HOOKS = {
    "resid_pre":      "blocks.0.hook_resid_pre",
    "ln1_normalized": "blocks.0.ln1.hook_normalized",
    "attn_out":       "blocks.0.hook_attn_out",
    "resid_mid":      "blocks.0.hook_resid_mid",
    "attn_pattern":   "blocks.0.attn.hook_pattern",
}

# SAE checkpoints (relative to EXP_DIR). Each value is a list of candidate paths
# tried in order; the first one that exists wins. "ln1" is optional — if none
# of its candidates exist, the z_ln1 encoding is skipped and downstream scripts
# fall back to per-head analysis using raw ln1.hook_normalized activations.
SAE_PATHS = {
    "pre":  [
        "recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer0.pt",
        "recreate_layer0/results/crosscoder_sae_layer0.pt",
    ],
    "mid":  [
        "recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer1.pt",
        "recreate_layer0/results/crosscoder_sae_layer1.pt",
    ],
    "ln1":  [
        "recreate_ln1/results/crosscoder_sae_layer0.pt",
    ],
}

# Ablation-sweep suppressor feature indices (from recreate_*/results/test_results.json).
SUPPRESSOR = {
    "pre_feature": 1359,
    "ln1_feature": 1412,
    "mid_feature": 171,
}


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def cache_layer0(
    model,
    tokens: torch.Tensor,             # (N, T) long
    chunk_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Forward-pass tokens with all block-0 hooks; return a dict of cached tensors."""
    device = next(model.parameters()).device
    keys = list(HOOKS)
    name_set = {HOOKS[k] for k in keys}
    outputs: dict[str, list[torch.Tensor]] = {k: [] for k in keys}

    for start in range(0, tokens.shape[0], chunk_size):
        batch = tokens[start : start + chunk_size].to(device)
        _, cache = model.run_with_cache(
            batch,
            return_type=None,
            names_filter=lambda n: n in name_set,
        )
        for k in keys:
            outputs[k].append(cache[HOOKS[k]].to(torch.float16).cpu())

    return {k: torch.cat(outputs[k], dim=0) for k in keys}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pre_sae", default=None)
    parser.add_argument("--mid_sae", default=None)
    parser.add_argument("--ln1_sae", default=None)
    parser.add_argument("--pre_feature", type=int, default=None)
    parser.add_argument("--mid_feature", type=int, default=None)
    parser.add_argument("--ln1_feature", type=int, default=None)
    parser.add_argument(
        "--output",
        default=str(HERE.parent / "results" / "layer0_cache.pt"),
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sae_paths = {
        "pre": [args.pre_sae] if args.pre_sae else SAE_PATHS["pre"],
        "mid": [args.mid_sae] if args.mid_sae else SAE_PATHS["mid"],
        "ln1": [args.ln1_sae] if args.ln1_sae else SAE_PATHS["ln1"],
    }
    suppressor = {
        "pre_feature": args.pre_feature if args.pre_feature is not None else SUPPRESSOR["pre_feature"],
        "mid_feature": args.mid_feature if args.mid_feature is not None else SUPPRESSOR["mid_feature"],
        "ln1_feature": args.ln1_feature if args.ln1_feature is not None else SUPPRESSOR["ln1_feature"],
    }

    print(f"[cache] device={device}")
    print(f"[cache] suppressor={suppressor}")
    print(f"[cache] loading sleeper model...")
    model = load_sleeper_model(device=device)
    print(
        f"[cache]   n_layers={model.cfg.n_layers} d_model={model.cfg.d_model} "
        f"n_heads={model.cfg.n_heads} d_head={model.cfg.d_head}"
    )

    print(f"[cache] loading paired dataset (seed={args.seed}, split={args.split})...")
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    pt = splits[args.split]
    N = pt.tokens.shape[0]
    print(f"[cache]   {args.split}: N={N}, dep_frac={pt.is_deployment.float().mean():.2f}")

    print(f"[cache] forward pass on {args.split}, caching block-0 hooks...")
    hooks = cache_layer0(model, pt.tokens, chunk_size=args.chunk_size)
    for k, v in hooks.items():
        print(f"[cache]   hook[{k}] {tuple(v.shape)} {v.dtype}")

    # Sanity: resid_mid ≈ resid_pre + attn_out (up to fp16 round-off).
    diff = (
        hooks["resid_mid"].float() - hooks["resid_pre"].float() - hooks["attn_out"].float()
    )
    max_abs = diff.abs().max().item()
    rel = diff.abs().mean().item() / hooks["resid_mid"].float().abs().mean().item()
    print(f"[cache] identity check: ||resid_mid - (resid_pre + attn_out)||_∞={max_abs:.3e}"
          f"  rel_mean={rel:.3e}")

    # ------------------------------------------------------------------
    # Encode through the three SAEs (pre, ln1, mid).
    # ------------------------------------------------------------------
    print(f"[cache] loading SAEs...")
    saes: dict[str, torch.nn.Module] = {}
    sae_configs: dict[str, dict] = {}
    for name, candidates in sae_paths.items():
        found: Path | None = None
        for rel in [c for c in candidates if c]:
            p = Path(rel)
            candidate_paths = [p] if p.is_absolute() else [p, EXP_DIR / rel]
            for candidate in candidate_paths:
                if candidate.exists():
                    found = candidate.resolve()
                    found_rel = str(found)
                    break
            if found is not None:
                break
        if found is None:
            print(f"[cache]   SAE[{name}] NOT FOUND (tried: {candidates}) — skipping")
            continue
        cc, cfg = load_crosscoder(found, device=device)
        saes[name] = cc
        sae_configs[name] = {
            "path": found_rel,
            "class_name": cfg["class_name"],
            "d_in": cfg["d_in"],
            "d_sae": cfg["d_sae"],
            "k_total": cfg["k_total"],
            "layer_hook": cfg.get("layer_hook"),
        }
        print(f"[cache]   SAE[{name}] path={found_rel} d_in={cfg['d_in']} d_sae={cfg['d_sae']} "
              f"k={cfg['k_total']} layer_hook={cfg.get('layer_hook')}")

    assert "mid" in saes, "SAE_mid is required"

    print(f"[cache] encoding activations through SAEs (post-TopK z)...")
    encodings = {}
    if "pre" in saes:
        encodings["z_pre"] = encode_all_sae(
            saes["pre"], hooks["resid_pre"].float(), chunk_size=256,
        )
    encodings["z_mid"] = encode_all_sae(
        saes["mid"], hooks["resid_mid"].float(), chunk_size=256,
    )
    if "ln1" in saes:
        encodings["z_ln1"] = encode_all_sae(
            saes["ln1"], hooks["ln1_normalized"].float(), chunk_size=256,
        )
    for k, v in encodings.items():
        print(f"[cache]   {k}: {tuple(v.shape)} {v.dtype}")

    # ------------------------------------------------------------------
    # Pre-activation logits (exact linear functional) for the target feature
    # in each SAE. These are what the decomposition reconstructs.
    # ------------------------------------------------------------------
    print(f"[cache] computing pre-activation logits for suppressor features...")

    @torch.no_grad()
    def pre_activation(cc, x: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Return W_enc[:, f] @ (x - b_dec) + b_enc[f], shape (N, T)."""
        dev = next(cc.parameters()).device
        w = cc.W_enc[:, feature_idx].to(dev)               # (d_in,)
        b_dec = cc.b_dec.to(dev)                           # (d_in,)
        b_enc_f = cc.b_enc[feature_idx].item()
        N_, T_, D_ = x.shape
        flat = x.reshape(N_ * T_, D_).to(dev, dtype=torch.float32)
        pre = (flat - b_dec) @ w + b_enc_f                 # (N*T,)
        return pre.reshape(N_, T_).cpu()

    pre_logits = {
        "mid_f_mid": pre_activation(
            saes["mid"], hooks["resid_mid"].float(), suppressor["mid_feature"],
        ),
    }
    if "pre" in saes:
        pre_logits["pre_f_pre"] = pre_activation(
            saes["pre"], hooks["resid_pre"].float(), suppressor["pre_feature"],
        )
    if "ln1" in saes:
        pre_logits["ln1_f_ln1"] = pre_activation(
            saes["ln1"], hooks["ln1_normalized"].float(), suppressor["ln1_feature"],
        )
    for k, v in pre_logits.items():
        print(f"[cache]   pre_logit[{k}]: {tuple(v.shape)}  "
              f"mean={v.mean():.3f}  std={v.std():.3f}")

    # ------------------------------------------------------------------
    # Save everything.
    # ------------------------------------------------------------------
    payload = {
        "meta": {
            "split": args.split,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "d_model": int(model.cfg.d_model),
            "n_heads": int(model.cfg.n_heads),
            "d_head": int(model.cfg.d_head),
            "hook_map": HOOKS,
            "sae_configs": sae_configs,
            "suppressor": suppressor,
            "identity_check_max_abs": max_abs,
        },
        "tokens": pt.tokens,
        "is_deployment": pt.is_deployment,
        "story_marker_pos": pt.story_marker_pos,
        "hooks": hooks,
        "encodings": encodings,
        "pre_logits": pre_logits,
    }
    torch.save(payload, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[cache] wrote {out_path} ({size_mb:.1f} MB)")

    # Meta sidecar so the cache's provenance is readable without loading the .pt.
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(payload["meta"], indent=2, default=str))
    print(f"[cache] wrote {meta_path}")


if __name__ == "__main__":
    main()
