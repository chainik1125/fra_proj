"""cell_runner.py — run ONE cadenza_sae_steer experiment cell to completion.

Launched in the BACKGROUND (nohup) by the autoresearch cron; the cron polls for
completion by checking whether this process is still alive. On exit it writes
`/workspace/jamie/orch/<name>.metrics.json` (with status ok|error) so the cron can
log the result. Designed to fail loudly into that JSON rather than hang.

Phase-1 'train' cell: build a streamed text mix at a target deployment fraction,
train ln1+resid_mid TopK SAEs at one layer with the VALIDATED recipe
(d_sae=32768, k=64, n_steps=12200 -> ~50M tokens, seq_len=128, lr=3e-4,
normalize=expected_average_only_in), then pull dead%/EV/L0 per cell from the
just-finished wandb run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback

ORCH = "/workspace/jamie/orch"
LLAMA_REPO = "/workspace/jamie/fra_proj_llama"
SLEEPERS_REPO = "/workspace/jamie/sleepers_repo"
PY = "/root/sleepers-venv/bin/python"
MIX_DIR = "/workspace/jamie/sae_mix"
SAE_DIR = "/workspace/jamie/saes/cadenza_mix"
WANDB_PROJECT = "cadenza-sae-steer"
WANDB_ENTITY = "jamiestephenson"


def build_mix(name: str, deployed_frac: float, total_rows: int) -> str:
    out = f"{MIX_DIR}/{name}"
    if os.path.exists(f"{out}/train-00000-of-00001.parquet"):
        return out
    subprocess.run(
        [PY, "scripts/cadenza_build_mix.py", "--out", out,
         "--deployed-frac", str(deployed_frac), "--total-rows", str(total_rows)],
        cwd=LLAMA_REPO, check=True)
    return out


def train(name: str, layer: int, hooks: list[str], seeds: list[int], mix_dir: str) -> str:
    out_dir = f"{SAE_DIR}/{name}/L{layer}"
    # saeguard/ on PYTHONPATH → Python auto-imports usercustomize.py, which guards
    # sae-lens's eval against the ce_loss_score/kl_div_score zero-division crash.
    guard_dir = f"{LLAMA_REPO}/scripts/saeguard"
    env = dict(
        os.environ,
        SAELENS_DATASET_PATH=mix_dir,
        SAELENS_N_CHECKPOINTS="0",
        SAELENS_DATA_SEED="0",
        WANDB_PROJECT=WANDB_PROJECT,
        WANDB_ENTITY=WANDB_ENTITY,
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=guard_dir + ((":" + os.environ["PYTHONPATH"]) if os.environ.get("PYTHONPATH") else ""),
    )
    # ONE hook per train_saes call. A 2-hook bank shares one LLM forward +
    # activation buffer (~8.6 GB/hook bf16) which OOMs an 80 GB H100 at d_in=4096.
    # Each hook gets its own forward+buffer; validated hyperparams unchanged
    # (chunk, don't shrink). Runs sequentially.
    for hook in hooks:
        subprocess.run(["rm", "-rf", "/tmp/saelens_ckpt"], check=False)
        cmd = [PY, "-m", "scripts.train_saes", "--model", "llama",
               "--layers", str(layer), "--hooks", hook,
               "--seeds", *[str(s) for s in seeds],
               "--sae_type", "topk", "--d_sae", "32768", "--k", "64",
               "--n_steps", "12200", "--batch_size", "4096", "--seq_len", "128",
               "--out_dir", out_dir]
        subprocess.run(cmd, cwd=SLEEPERS_REPO, env=env, check=True)
    return out_dir


def pull_metrics(n_runs: int) -> dict:
    """dead_features / EV / l0 per cell from the most-recent n_runs (one per hook).
    Cell keys (L9_ln1/s0/... vs L9_resid_mid/s0/...) don't collide, so merge."""
    import wandb
    api = wandb.Api()
    runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                         order="-created_at", per_page=max(3, n_runs + 1)))[:n_runs]
    out: dict = {"_runs": [r.id for r in runs]}
    for r in runs:
        for k in list(r.summary.keys()):
            kl = k.lower()
            if any(t in kl for t in ("dead_features", "explained_variance", "/l0", "mse")):
                try:
                    out[k] = round(float(r.summary[k]), 5)
                except (TypeError, ValueError):
                    pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="cell id, e.g. dep05_L9")
    ap.add_argument("--deployed-frac", type=float, required=True)
    ap.add_argument("--total-rows", type=int, default=100_000)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hooks", nargs="+", default=["ln1", "resid_mid"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="1 seed for a mix screen; expand the winner later")
    a = ap.parse_args()

    os.makedirs(ORCH, exist_ok=True)
    res: dict = {"name": a.name, "deployed_frac": a.deployed_frac, "layer": a.layer,
                 "hooks": a.hooks, "seeds": a.seeds, "total_rows": a.total_rows,
                 "started": time.time()}
    try:
        mix = build_mix(a.name, a.deployed_frac, a.total_rows)
        res["out_dir"] = train(a.name, a.layer, a.hooks, a.seeds, mix)
        res["metrics"] = pull_metrics(len(a.hooks))
        res["status"] = "ok"
    except Exception as e:  # fail loudly into the json so the cron sees it
        res["status"] = "error"
        res["error"] = repr(e)
        res["trace"] = traceback.format_exc()[-2000:]
    res["ended"] = time.time()
    res["minutes"] = round((res["ended"] - res["started"]) / 60, 1)
    with open(f"{ORCH}/{a.name}.metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[cell_runner] {a.name} -> {res['status']} ({res['minutes']} min)")


if __name__ == "__main__":
    main()
