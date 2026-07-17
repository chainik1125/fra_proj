"""Run the one-step gradient-projection experiment on Modal GPUs.

Maps over all (prior, seed) combinations in one launch. Each container pretrains the
headline model on its prior, fits base probes, applies one fine-tuning update
(sgd_accum / sgd_quarter / adam_first), and returns the decoded-shift CSV. The local
entrypoint combines everything into
experiment_folders/em_afp_simpler_codex_auto/results_grad_projection/combined_grad_projection.csv
and prints the per-prior summary.

Usage:
    uv run modal run cloud/modal_grad_projection.py --smoke
    uv run modal run cloud/modal_grad_projection.py --seeds 0,1,2
"""

from __future__ import annotations

import pathlib

import modal


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_grad_projection"

DEFAULT_PI0S = (
    "0.010,0.040,0.490,0.460",
    "0.015,0.035,0.485,0.465",
    "0.025,0.025,0.475,0.475",
    "0.035,0.015,0.465,0.485",
    "0.040,0.010,0.460,0.490",
)

app = modal.App("special-sfp-grad-projection")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "matplotlib")
    .add_local_dir(str(ROOT / "bag_moments"), "/work/bag_moments")
    .add_local_dir(str(ROOT / "experiments"), "/work/experiments")
)


# max_inputs=1: the experiment module reads its config from env at import time, so a
# reused container would silently keep the first item's seed/prior. One input per container.
@app.function(gpu="A10G", image=image, timeout=3600, memory=16384, max_inputs=1)
def run_remote(params: dict) -> tuple[dict, bytes]:
    import io
    import json
    import os
    import sys
    import zipfile
    from pathlib import Path

    sys.path.insert(0, "/work")
    out = Path("/tmp/grad_projection")
    os.environ["SPECIAL_SFP_GRAD_OUT"] = str(out)
    os.environ["BAG_DEVICE"] = "cuda"
    for key, value in params.items():
        os.environ[key] = str(value)

    from experiments.special_sfp_grad_projection import main

    main()
    seed = params.get("SPECIAL_SFP_GRAD_SEED", 0)
    metadata = json.loads((out / f"metadata_seed{seed}.json").read_text())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out).as_posix())
    return metadata, buf.getvalue()


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2",
    pi0s: str = "",
    smoke: bool = False,
    base_steps: int = 0,
    out: str = "",
):
    import json
    import zipfile
    from io import BytesIO

    import pandas as pd

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    pi0_list = [p.strip() for p in pi0s.split(";") if p.strip()] if pi0s else list(DEFAULT_PI0S)

    base_params: dict[str, str | int] = {}
    if smoke:
        base_params["SPECIAL_SFP_GRAD_SMOKE"] = "1"
    if base_steps:
        base_params["SPECIAL_SFP_GRAD_BASE_STEPS"] = base_steps

    out_dir = pathlib.Path(out) if out else DEFAULT_OUT
    if smoke:
        out_dir = out_dir.parent / (out_dir.name + "_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [(pi0, seed) for pi0 in pi0_list for seed in seed_list]
    param_list = [
        {**base_params, "SPECIAL_SFP_PROBE_PI0": pi0, "SPECIAL_SFP_GRAD_SEED": seed}
        for pi0, seed in combos
    ]
    results = list(run_remote.map(param_list))

    frames = []
    for (pi0, seed), (metadata, blob) in zip(combos, results):
        tag = pi0.replace(",", "_").replace(".", "p")
        sub = out_dir / f"pi0_{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(blob), "r") as zf:
            zf.extractall(sub)
        frames.append(pd.read_csv(sub / f"grad_projection_seed{seed}.csv"))
        print(f"pi0={pi0} seed {seed}: elapsed {metadata['elapsed_s']:.0f}s")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "combined_grad_projection.csv", index=False)
    (out_dir / "modal_run.json").write_text(
        json.dumps({"seeds": seed_list, "pi0s": pi0_list, "params": base_params}, indent=2) + "\n",
        encoding="utf-8",
    )

    last_layer = combined["layer"].max()
    summary = (
        combined[(combined["layer"] == last_layer)]
        .groupby(["pi0", "method", "context"])[["z_P_M", "z_P_D", "z_det", "z_MD", "z_MO"]]
        .mean()
        .round(3)
    )
    print(summary.to_string())
    print(f"wrote {out_dir}")
