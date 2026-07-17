"""Run the special-SFP belief-probe factorization experiment on Modal GPUs.

Ships bag_moments/ + experiments/ into a CUDA image and executes
`experiments/special_sfp_probe_factorization.py` remotely, one container per
seed in parallel. Each remote run returns a zip of its output directory, which
the local entrypoint saves under
experiment_folders/em_afp_simpler_codex_auto/results_probe_factorization/seed{N}
and concatenates into combined CSVs.

Also supports the non-product-prior control via --pi0 (comma-separated MD,MO,AD,AO).

Keep P(M) and P(D) fixed when choosing a control prior (marginal confound otherwise):
the family pi = (a, 0.05-a, 0.5-a, 0.45+a) holds P(M)=0.05, P(D)=0.5 for any a.

Usage:
    uv run modal run cloud/modal_probe_factorization.py --smoke
    uv run modal run cloud/modal_probe_factorization.py --seeds 0,1,2
    uv run modal run cloud/modal_probe_factorization.py --seeds 0,1,2 --probe-n 8192 --pi0 0.040,0.010,0.460,0.490
"""

from __future__ import annotations

import pathlib

import modal


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = (
    ROOT
    / "experiment_folders"
    / "em_afp_simpler_codex_auto"
    / "results_probe_factorization"
)

app = modal.App("special-sfp-probe-factorization")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "matplotlib")
    .add_local_dir(str(ROOT / "bag_moments"), "/work/bag_moments")
    .add_local_dir(str(ROOT / "experiments"), "/work/experiments")
)
# Results are ALSO committed to this volume so a local disconnect cannot lose them
# (.map() results in detached apps are canceled when the local caller dies). Recover:
#   uv run --with modal modal volume get probe-outputs <tag>/seed<N>.zip
out_vol = modal.Volume.from_name("probe-outputs", create_if_missing=True)


# max_inputs=1: the experiment module reads its config from env at import time, so a
# reused container would silently keep the first item's seed/prior. One input per container.
@app.function(gpu="A10G", image=image, timeout=3600, memory=16384, max_inputs=1, volumes={"/outvol": out_vol})
def run_remote(params: dict) -> tuple[dict, bytes]:
    import io
    import json
    import os
    import sys
    import zipfile
    from pathlib import Path

    sys.path.insert(0, "/work")
    out = Path("/tmp/probe_factorization")
    os.environ["SPECIAL_SFP_PROBE_OUT"] = str(out)
    os.environ["BAG_DEVICE"] = "cuda"
    for key, value in params.items():
        os.environ[key] = str(value)

    from experiments.special_sfp_probe_factorization import main

    main()
    metadata = json.loads((out / "metadata.json").read_text())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out).as_posix())
    blob = buf.getvalue()

    tag = params.get("OUTPUT_TAG", "untagged")
    seed = params.get("SPECIAL_SFP_PROBE_SEED", 0)
    vol_path = Path(f"/outvol/{tag}")
    vol_path.mkdir(parents=True, exist_ok=True)
    (vol_path / f"seed{seed}.zip").write_bytes(blob)
    out_vol.commit()
    return metadata, blob


@app.local_entrypoint()
def main(
    seeds: str = "0",
    smoke: bool = False,
    base_steps: int = 0,
    ft_lr: float = 0.0,
    ft_steps: int = 0,
    checkpoints: str = "",
    pi0: str = "",
    probe_n: int = 0,
    rollout_n: int = 0,
    head_only: bool = False,
    prompt_k: int = 0,
    prompt_len: int = 0,
    prompt_nper: int = 0,
    out: str = "",
):
    import json
    import zipfile
    from io import BytesIO

    import pandas as pd

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    base_params: dict[str, str | int | float] = {}
    if smoke:
        base_params["SPECIAL_SFP_PROBE_SMOKE"] = "1"
    if base_steps:
        base_params["SPECIAL_SFP_PROBE_BASE_STEPS"] = base_steps
    if ft_lr:
        base_params["SPECIAL_SFP_PROBE_FT_LR"] = ft_lr
    if pi0:
        base_params["SPECIAL_SFP_PROBE_PI0"] = pi0
    if probe_n:
        base_params["SPECIAL_SFP_PROBE_N"] = probe_n
    if rollout_n:
        base_params["SPECIAL_SFP_PROBE_ROLLOUT_N"] = rollout_n
    if ft_steps:
        base_params["SPECIAL_SFP_PROBE_FT_STEPS"] = ft_steps
    if checkpoints:
        base_params["SPECIAL_SFP_PROBE_CHECKPOINTS"] = checkpoints
    if head_only:
        base_params["SPECIAL_SFP_PROBE_HEAD_ONLY"] = "1"
    if prompt_k:
        base_params["SPECIAL_SFP_PROBE_PROMPT_K"] = prompt_k
    if prompt_len:
        base_params["SPECIAL_SFP_PROBE_PROMPT_LEN"] = prompt_len
    if prompt_nper:
        base_params["SPECIAL_SFP_PROBE_PROMPT_NPER"] = prompt_nper

    out_dir = pathlib.Path(out) if out else DEFAULT_OUT
    if pi0 and not out:
        tag = pi0.replace(",", "_").replace(".", "p")
        out_dir = out_dir.parent / (out_dir.name + f"_pi0_{tag}")
    if head_only and not out:
        out_dir = out_dir.parent / (out_dir.name + "_headonly")
    if smoke:
        out_dir = out_dir.parent / (out_dir.name + "_smoke")

    base_params["OUTPUT_TAG"] = out_dir.name
    param_list = [{**base_params, "SPECIAL_SFP_PROBE_SEED": seed} for seed in seed_list]
    results = list(run_remote.map(param_list))

    frames: dict[str, list[pd.DataFrame]] = {"probe_metrics": [], "drift_decomposition": [], "prompt_decoding": []}
    for seed, (metadata, blob) in zip(seed_list, results):
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(blob), "r") as zf:
            zf.extractall(seed_dir)
        print(f"seed {seed}: elapsed {metadata['elapsed_s']:.0f}s, device {metadata.get('device')}")
        for name in frames:
            df = pd.read_csv(seed_dir / f"{name}.csv")
            df.insert(0, "seed", seed)
            frames[name].append(df)

    for name, dfs in frames.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(out_dir / f"combined_{name}.csv", index=False)
    (out_dir / "modal_run.json").write_text(
        json.dumps({"seeds": seed_list, "params": base_params}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")
