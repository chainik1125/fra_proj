"""Main driver for the Fisher-guided clean-recovery POC.

Two modes:

  --mode expA   Re-evaluate the existing 1-D α-sweep under the new
                Fisher-path-length metric L_F.  K=1 in each control space;
                emits per-α curves (J_clean, CRF, F_aa, L_F).

  --mode expB   Greedy diagonal-Fisher selection across K candidate
                features in each control space.  Emits a trajectory of
                (step, selected feature, J, CRF, step_jsd, accepted).

Both modes run independently for the FRA-OV→OV space and the resid-mid
conventional space.  Outputs land under results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(HERE))

from sleeper_utils import (  # noqa: E402
    BASE_MODEL_NAME,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from run_ablation_sweep import load_crosscoder  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from control_space import FRAOVControlSpace, ResidMidControlSpace, unsteered_logits  # noqa: E402
from fisher_utils import (  # noqa: E402
    fisher_path_length_1d,
    greedy_fisher_clean_recovery,
    jsd_bits,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(device: str) -> HookedTransformer:
    m = HookedTransformer.from_pretrained(BASE_MODEL_NAME, device=device)
    m.eval()
    return m


def load_top_features(ranking_path: Path, k: int) -> list[int]:
    data = json.loads(ranking_path.read_text())
    return list(map(int, data["top_indices"][:k]))


def prediction_mask_from_markers(seq_len: int, marker_pos: torch.Tensor) -> torch.Tensor:
    """Positions in [0, seq_len-1) that predict tokens after `Story:`."""
    source_pos = torch.arange(seq_len - 1).unsqueeze(0)
    return source_pos >= marker_pos.unsqueeze(1)


def select_batch(pt, batch_size: int, deployment_only: bool):
    if deployment_only:
        rows = torch.where(pt.is_deployment)[0]
    else:
        rows = torch.arange(pt.tokens.shape[0])
    rows = rows[:batch_size]
    return (
        pt.tokens[rows],
        pt.is_deployment[rows],
        pt.story_marker_pos[rows],
    )


def build_clean_logits(base_model, tokens, device):
    with torch.no_grad():
        return base_model(tokens.to(device), return_type="logits")


def make_predicting_forward(space, T_pred: int):
    """Wrap a control space's forward_logits to slice off the trailing position.

    `prediction_mask_from_markers` returns shape [B, T-1] (positions that
    predict post-marker tokens). Logits at position T-1 have no target, so
    we drop them everywhere — keeps shapes aligned with the mask.
    """
    def _forward(theta):
        logits = space.forward_logits(theta)
        return logits[:, :T_pred, :]
    return _forward


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def run_exp_a(cfg, args, sleeper_model, base_model, batch, mask):
    """Experiment A: K=1 alpha sweep in each control space, with Fisher path length."""
    tokens, _is_dep, marker_pos = batch
    device = next(sleeper_model.parameters()).device

    prompt_mask_full = prompt_mask_from_markers(args.seq_len, marker_pos)
    clean_logits_full = build_clean_logits(base_model, tokens, device)
    T_pred = mask.shape[1]
    clean_logits = clean_logits_full[:, :T_pred, :]

    # === FRA OV→OV (K=1) ===
    sae_ov, _ = load_crosscoder(EXP_ROOT / cfg["fra_ov"]["sae_path"], device=device)
    feat_id_ov = int(cfg["exp_a"]["fra_ov_feature"])
    space_ov = FRAOVControlSpace(sleeper_model, sae_ov, [feat_id_ov], tokens, prompt_mask_full)
    fwd_ov = make_predicting_forward(space_ov, T_pred)
    print(f"[expA] FRA OV space: feature={feat_id_ov}, K=1")
    out_ov = fisher_path_length_1d(
        fwd_ov,
        cfg["exp_a"]["alphas"],
        clean_logits,
        mask,
        eps_fd=cfg["exp_b"]["eps_fd"],
    )

    # === resid-mid (K=1) ===
    sae_rm, _ = load_crosscoder(EXP_ROOT / cfg["resid_mid"]["sae_path"], device=device)
    feat_id_rm = int(cfg["exp_a"]["resid_mid_feature"])
    space_rm = ResidMidControlSpace(sleeper_model, sae_rm, [feat_id_rm], tokens, prompt_mask_full)
    fwd_rm = make_predicting_forward(space_rm, T_pred)
    print(f"[expA] resid-mid space: feature={feat_id_rm}, K=1")
    out_rm = fisher_path_length_1d(
        fwd_rm,
        cfg["exp_a"]["alphas"],
        clean_logits,
        mask,
        eps_fd=cfg["exp_b"]["eps_fd"],
    )

    return {
        "fra_ov": {"feature": feat_id_ov, **out_ov},
        "resid_mid": {"feature": feat_id_rm, **out_rm},
    }


def run_exp_b(cfg, args, sleeper_model, base_model, batch, mask):
    """Experiment B: greedy diagonal Fisher in each control space."""
    tokens, _is_dep, marker_pos = batch
    device = next(sleeper_model.parameters()).device
    prompt_mask_full = prompt_mask_from_markers(args.seq_len, marker_pos)
    clean_logits_full = build_clean_logits(base_model, tokens, device)
    T_pred = mask.shape[1]
    clean_logits = clean_logits_full[:, :T_pred, :]

    results: dict[str, Any] = {}

    # === FRA OV→OV ===
    sae_ov, _ = load_crosscoder(EXP_ROOT / cfg["fra_ov"]["sae_path"], device=device)
    ov_feats = load_top_features(EXP_ROOT / cfg["fra_ov"]["ranking_path"], cfg["fra_ov"]["K"])
    print(f"[expB] FRA OV: K={len(ov_feats)} features (top of ranking)")
    space_ov = FRAOVControlSpace(sleeper_model, sae_ov, ov_feats, tokens, prompt_mask_full)
    fwd_ov = make_predicting_forward(space_ov, T_pred)
    t0 = time.time()
    theta_ov, traj_ov = greedy_fisher_clean_recovery(
        fwd_ov,
        K=space_ov.K,
        clean_logits=clean_logits,
        mask=mask,
        num_steps=cfg["exp_b"]["num_steps"],
        eps_fd=cfg["exp_b"]["eps_fd"],
        target_step_jsd_bits=cfg["exp_b"]["target_step_jsd_bits"],
        delta_cap=cfg["exp_b"]["delta_cap"],
        device=device,
    )
    print(f"[expB] FRA OV done in {time.time()-t0:.1f}s, steps={len(traj_ov)}")
    results["fra_ov"] = {
        "feature_ids": ov_feats,
        "theta": theta_ov.cpu().tolist(),
        "trajectory": [asdict(s) for s in traj_ov],
    }

    # === resid-mid ===
    sae_rm, _ = load_crosscoder(EXP_ROOT / cfg["resid_mid"]["sae_path"], device=device)
    rm_feats = load_top_features(EXP_ROOT / cfg["resid_mid"]["ranking_path"], cfg["resid_mid"]["K"])
    print(f"[expB] resid-mid: K={len(rm_feats)} features (top of ranking)")
    space_rm = ResidMidControlSpace(sleeper_model, sae_rm, rm_feats, tokens, prompt_mask_full)
    fwd_rm = make_predicting_forward(space_rm, T_pred)
    t0 = time.time()
    theta_rm, traj_rm = greedy_fisher_clean_recovery(
        fwd_rm,
        K=space_rm.K,
        clean_logits=clean_logits,
        mask=mask,
        num_steps=cfg["exp_b"]["num_steps"],
        eps_fd=cfg["exp_b"]["eps_fd"],
        target_step_jsd_bits=cfg["exp_b"]["target_step_jsd_bits"],
        delta_cap=cfg["exp_b"]["delta_cap"],
        device=device,
    )
    print(f"[expB] resid-mid done in {time.time()-t0:.1f}s, steps={len(traj_rm)}")
    results["resid_mid"] = {
        "feature_ids": rm_feats,
        "theta": theta_rm.cpu().tolist(),
        "trajectory": [asdict(s) for s in traj_rm],
    }

    # Sanity: also record baseline JSD at θ=0 in each space.
    with torch.no_grad():
        u_logits = unsteered_logits(sleeper_model, tokens)[:, :T_pred, :]
        results["baseline_jsd_bits"] = float(jsd_bits(u_logits, clean_logits, mask).item())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--mode", choices=["expA", "expB", "both"], default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config dataset.seed; controls the paired-dataset sample.",
    )
    parser.add_argument("--output_dir", default=str(HERE / "results"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = pick_device(args.device)
    args.seq_len = args.seq_len or cfg["dataset"]["seq_len"]
    if args.seed is not None:
        cfg["dataset"]["seed"] = int(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fisher_poc] device={device}")
    print(f"[fisher_poc] loading sleeper + base...")
    sleeper_model = load_sleeper_model(device=device)
    base_model = load_base_model(device=device)

    print(f"[fisher_poc] loading paired dataset (split={cfg['dataset']['split']})...")
    splits = load_paired_dataset(
        tokenizer=sleeper_model.tokenizer,
        n_train=cfg["dataset"]["n_train"],
        n_val=cfg["dataset"]["n_val"],
        n_test=cfg["dataset"]["n_test"],
        seq_len=args.seq_len,
        seed=cfg["dataset"]["seed"],
    )
    pt = splits[cfg["dataset"]["split"]]
    batch = select_batch(
        pt,
        batch_size=cfg["dataset"]["batch_size"],
        deployment_only=cfg["dataset"]["deployment_only"],
    )
    tokens, is_dep, marker_pos = batch
    tokens = tokens.to(device)
    marker_pos = marker_pos.to(device)
    batch = (tokens, is_dep.to(device), marker_pos)

    pred_mask = prediction_mask_from_markers(args.seq_len, marker_pos)
    print(f"[fisher_poc] batch: B={tokens.shape[0]} T={tokens.shape[1]} "
          f"masked_positions={int(pred_mask.sum().item())}")

    seed_tag = f"seed{cfg['dataset']['seed']}"

    if args.mode in ("expA", "both"):
        print("[fisher_poc] === Experiment A: 1-D α paths under L_F ===")
        out_a = run_exp_a(cfg, args, sleeper_model, base_model, batch, pred_mask)
        out_a["seed"] = int(cfg["dataset"]["seed"])
        write_json(out_dir / f"expA_{seed_tag}.json", out_a)
        print(f"[fisher_poc] wrote {out_dir / f'expA_{seed_tag}.json'}")

    if args.mode in ("expB", "both"):
        print("[fisher_poc] === Experiment B: greedy diagonal Fisher ===")
        out_b = run_exp_b(cfg, args, sleeper_model, base_model, batch, pred_mask)
        out_b["seed"] = int(cfg["dataset"]["seed"])
        write_json(out_dir / f"expB_{seed_tag}.json", out_b)
        print(f"[fisher_poc] wrote {out_dir / f'expB_{seed_tag}.json'}")

    print("[fisher_poc] done.")


if __name__ == "__main__":
    main()
