"""Run the toy_ec corrective-transition sweep on Modal GPUs, seeds in parallel.

Usage (from the worktree root):
  uv run modal run cloud/modal_toy_ec.py                       # 4 seeds, full grid
  uv run modal run cloud/modal_toy_ec.py --seeds 0 --smoke     # 1 fast smoke seed

Outputs: toy_ec/outputs/ec_sweep/ec_sweep_seed{N}.pkl (one per seed).
"""
import pathlib
import pickle

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("toy-ec-sweep")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "numpy", "pyyaml", "tqdm", "jax",
        "transformer-lens",
        "git+https://github.com/Astera-org/simplexity.git@dev",
    )
    .add_local_dir(str(ROOT / "toy_ec"), "/work/toy_ec")
)


@app.function(gpu="A10G", image=image, timeout=3 * 3600)
def run_seed(seed: int, params: dict) -> bytes:
    import sys
    sys.path.insert(0, "/work/toy_ec")
    from run_ec_sweep import run_sweep

    res = run_sweep(seed=seed, **params)
    return pickle.dumps(res)


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3", smoke: bool = False, d_models: str = "64",
         control: bool = True, dup_caps: str = "", fracs: str = "",
         pivot_ks: str = "", tag: str = ""):
    seed_list = [int(s) for s in seeds.split(",")]
    dm_list = [int(d) for d in d_models.split(",")]
    base_params = dict(
        fracs=[0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5],
        control_fracs=[0.05, 0.1, 0.25, 0.5] if control else [],
        pretrain_steps=5000,
        ft_steps=2000,
        n_gen=20,
    )
    if dup_caps:
        base_params["dup_caps"] = [int(x) for x in dup_caps.split(",")]
    if pivot_ks:
        base_params["pivot_ks"] = [int(x) for x in pivot_ks.split(",")]
    if fracs:
        base_params["fracs"] = [float(x) for x in fracs.split(",")]
    if smoke:
        base_params.update(pretrain_steps=200, ft_steps=50, n_gen=2,
                           fracs=[0.0, 0.5], control_fracs=[0.5])

    out_dir = ROOT / "toy_ec" / "outputs" / "ec_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(seed, dict(base_params, d_model=dm)) for dm in dm_list for seed in seed_list]
    for (seed, params), blob in zip(jobs, run_seed.starmap(jobs)):
        dm = params["d_model"]
        suffix = (f"_d{dm}" if dm != 64 else "") + (f"_{tag}" if tag else "")
        path = out_dir / f"ec_sweep_seed{seed}{suffix}.pkl"
        path.write_bytes(blob)
        res = pickle.loads(blob)
        print(f"seed {seed} d={dm}: wall {res['wall_seconds']:.0f}s, "
              f"{len(res['conditions'])} conditions -> {path}")
    print("done")
