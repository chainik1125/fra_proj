"""
Simple training report - saves data to JSON and loss curve SVG.

Usage:
    from report import save_report
    save_report(results, cfg, "my_run")
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "small_outputs"


def save_report(results: dict, cfg, run_name: str = None) -> Path:
    """Save results to JSON and generate loss curve SVG."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON with results and config
    json_path = OUTPUT_DIR / f"{run_name}.json"
    data = {
        "results": results,
        "cfg": {
            "process_name": cfg.process_name,
            "process_params": cfg.process_params,
            "processes": cfg.processes,
            "num_steps": cfg.num_steps,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
            "n_ctx": cfg.n_ctx,
            "device": cfg.device,
            "seed": cfg.seed,
        }
    }
    json_path.write_text(json.dumps(data, indent=2))

    # Generate loss curve SVG
    history = results["history"]
    steps = [d["step"] for d in history]
    train_loss = [d["train_loss"] for d in history]
    eval_steps = [d["step"] for d in history if "eval_loss" in d]
    eval_loss = [d["eval_loss"] for d in history if "eval_loss" in d]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label="Train")
    plt.plot(eval_steps, eval_loss, "o-", label="Eval")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    svg_path = OUTPUT_DIR / f"{run_name}.svg"
    plt.savefig(svg_path, format="svg")
    plt.close()

    print(f"Saved: {json_path}")
    print(f"Saved: {svg_path}")
    return svg_path
