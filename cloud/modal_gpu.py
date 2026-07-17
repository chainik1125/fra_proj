"""
Run the multi-seed robustness experiment at FULL headline config on a Modal GPU (HTTPS-only,
so this also works from the locked-down cloud sandbox). Drives the committed
experiments/exp_multiseed_robustness.py functions with device='cuda' and L=200 — no edit to
the committed script.

Run:  cd <worktree> && uv run modal run cloud/modal_gpu.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("bag-multiseed-gpu")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(str(ROOT / "bag_moments"), "/work/bag_moments")
    .add_local_dir(str(ROOT / "experiments"), "/work/experiments")
)


@app.function(gpu="A10G", image=image, timeout=3600)
def run_full():
    import os, sys, importlib.util
    # full headline config (env read at module import time)
    os.environ.update({"BAG_ROOT": "/work", "BAG_OUT": "/tmp/out",
                       "C1_STEPS": "6000", "C2_STEPS": "6000",
                       "C1_BATCH": "256", "C2_BATCH": "256",
                       "C1_NSEED": "5", "C2_NSEED": "3", "C2_NS": "3,5,7"})
    sys.path.insert(0, "/work")
    spec = importlib.util.spec_from_file_location("ms", "/work/experiments/exp_multiseed_robustness.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import numpy as np, torch
    m.L = 200  # full sequence length (headline); module uses this global at call time
    dev = "cuda"
    c1 = [m.run_c1_seed(s, dev) for s in m.C1_SEEDS]
    c2 = {n: [m.run_c2_seed(s, n, dev) for s in m.C2_SEEDS] for n in m.C2_NS}

    kls = [r["kl"] for r in c1]; r2s = [r["best_r2"] for r in c1]; cr2 = [r["count_r2"] for r in c1]
    c1sum = {"by_seed": c1, "n_seeds": len(m.C1_SEEDS),
             "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
             "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
             "count_r2_mean": float(np.mean(cr2)), "count_r2_std": float(np.std(cr2)),
             "config": {"L": 200, "steps": 6000, "batch": 256}}
    c2sum = {}
    for n in m.C2_NS:
        rows = c2[n]
        kk = [r["kl"] for r in rows]; me = [r["model_logical_err"] for r in rows]
        be = [r["bayes_logical_err"] for r in rows]; pe = [r["phys_bayes_err"] for r in rows]
        c2sum[n] = {"by_seed": rows, "n_seeds": len(m.C2_SEEDS),
                    "kl_mean": float(np.mean(kk)), "kl_std": float(np.std(kk)),
                    "model_logical_err_mean": float(np.mean(me)), "model_logical_err_std": float(np.std(me)),
                    "bayes_logical_err_mean": float(np.mean(be)), "phys_bayes_err_mean": float(np.mean(pe))}
    return {"device": torch.cuda.get_device_name(0), "config": {"L": 200, "steps": 6000, "batch": 256},
            "c1": c1sum, "c2": c2sum}


@app.local_entrypoint()
def main():
    import torch, pathlib
    res = run_full.remote()
    print("DEVICE:", res["device"])
    c1 = res["c1"]
    print(f"C1 ({c1['n_seeds']} seeds): KL {c1['kl_mean']:.5f} ± {c1['kl_std']:.5f} | "
          f"z R² {c1['r2_mean']:.4f} ± {c1['r2_std']:.4f} | count→z R² {c1['count_r2_mean']:.4f}")
    for n, d in res["c2"].items():
        print(f"C2 n={n} ({d['n_seeds']} seeds): KL {d['kl_mean']:.5f}±{d['kl_std']:.5f} | "
              f"model_err {d['model_logical_err_mean']:.4f}±{d['model_logical_err_std']:.4f} "
              f"| bayes {d['bayes_logical_err_mean']:.4f} | phys {d['phys_bayes_err_mean']:.4f}")
    pathlib.Path("results").mkdir(exist_ok=True)
    torch.save(res["c1"], "results/c1_multiseed_full.pt")
    torch.save(res["c2"], "results/c2_multiseed_full.pt")
    print("saved results/c1_multiseed_full.pt + results/c2_multiseed_full.pt")
