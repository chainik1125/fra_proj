"""Reproduce the TinyStories sleeper TXC-vs-MLC-vs-SAE benchmark.

Routes every artifact (models, JSONs, plots, RESULTS.md, run.log) into the
adjacent `results/` directory so the complete run is self-contained.

Usage:
    python reproduce.py                  # uses ./config.yaml, full pipeline
    python reproduce.py --skip harvest   # re-use cached activations
    python reproduce.py my_config.yaml --skip harvest train   # re-plot only
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
EXPERIMENT_DIR = HERE.parent


def run(cmd: list[str], log_file) -> None:
    line = "$ " + " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[reproduce] {line}", flush=True)
    log_file.write(f"\n{line}\n")
    log_file.flush()
    # tee stdout/stderr to both the console and run.log
    proc = subprocess.Popen(
        cmd, cwd=EXPERIMENT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log_file.write(line)
        log_file.flush()
    proc.wait()
    if proc.returncode != 0:
        log_file.write(f"[reproduce] step failed (exit {proc.returncode})\n")
        sys.exit(f"[reproduce] step failed with exit code {proc.returncode}")


FILE_CATEGORIES = [
    ("Summary", ["RESULTS.md", "pareto_asr_vs_utility.png"]),
    ("Trained crosscoders", ["crosscoder_*.pt"]),
    ("Feature rankings", ["feature_rankings_*.pt", "feature_rankings_*.json"]),
    ("Sweep outputs", ["val_sweep_*.json", "test_results.json"]),
    ("Metadata", ["harvest_meta.json", "train_meta.json"]),
    ("Cache (regenerable, may be large)", ["activations_cache.pt", "tokens_cache.pt"]),
    ("Logs", ["run.log"]),
]


def write_manifest(results_dir: Path, config_path: Path) -> None:
    lines = [
        "# recreate/results/ manifest",
        "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by `reproduce.py`.",
        f"Config: `{config_path.name}`",
        "",
    ]
    seen: set[str] = set()
    for category, patterns in FILE_CATEGORIES:
        matches: list[Path] = []
        for pat in patterns:
            matches.extend(sorted(results_dir.glob(pat)))
        matches = [p for p in matches if p.name not in seen]
        if not matches:
            continue
        lines.append(f"## {category}")
        lines.append("")
        for p in matches:
            seen.add(p.name)
            size = p.stat().st_size
            pretty = _pretty_size(size)
            lines.append(f"- `{p.name}` — {pretty}")
        lines.append("")
    # Leftovers not matched by any pattern
    others = sorted(p for p in results_dir.iterdir() if p.name not in seen and p.name != "MANIFEST.md")
    if others:
        lines.append("## Other")
        lines.append("")
        for p in others:
            if p.is_file():
                lines.append(f"- `{p.name}` — {_pretty_size(p.stat().st_size)}")
            else:
                lines.append(f"- `{p.name}/` (directory)")
        lines.append("")
    (results_dir / "MANIFEST.md").write_text("\n".join(lines))


def _pretty_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024
    return f"{n:,.1f} GB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(HERE / "config.yaml"))
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["harvest", "train", "sweep", "plot"],
                        help="Skip pipeline steps (their outputs must already be present in results/).")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())

    env = cfg.get("env", {})
    harvest = cfg["harvest"]
    train = cfg["train"]
    sweep = cfg["sweep"]
    device = env.get("device")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "run.log"
    data_dir = str(RESULTS_DIR)       # all scripts write/read here

    def _device_flag() -> list[str]:
        return ["--device", device] if device else []

    with log_path.open("a") as log:
        log.write(f"\n======= reproduce.py run @ {dt.datetime.now().isoformat(timespec='seconds')} =======\n")
        log.write(f"config: {config_path}\n")
        log.write(f"skip: {args.skip}\n")

        if "harvest" not in args.skip:
            run([
                sys.executable, "harvest_activations.py",
                "--n_train", str(harvest["n_train"]),
                "--n_val", str(harvest["n_val"]),
                "--n_test", str(harvest["n_test"]),
                "--seq_len", str(harvest["seq_len"]),
                "--chunk_size", str(harvest["chunk_size"]),
                "--seed", str(harvest["seed"]),
                "--output_dir", data_dir,
                *_device_flag(),
            ], log)

        if "train" not in args.skip:
            run([
                sys.executable, "train_crosscoders.py",
                "--d_sae", str(train["d_sae"]),
                "--k_total", str(train["k_total"]),
                "--T", str(train["T"]),
                "--n_steps", str(train["n_steps"]),
                "--batch_size", str(train["batch_size"]),
                "--lr", str(train["lr"]),
                "--normalize_every", str(train["normalize_every"]),
                "--print_every", str(train["print_every"]),
                "--seed", str(train["seed"]),
                "--input_dir", data_dir,
                "--output_dir", data_dir,
                *_device_flag(),
            ], log)

        if "sweep" not in args.skip:
            run([
                sys.executable, "run_ablation_sweep.py",
                "--top_k", str(sweep["top_k"]),
                "--stage2_keep", str(sweep["stage2_keep"]),
                "--alphas", *[str(a) for a in sweep["alphas"]],
                "--delta_util", str(sweep["delta_util"]),
                "--gen_tokens", str(sweep["gen_tokens"]),
                "--encode_chunk_size", str(sweep["encode_chunk_size"]),
                "--input_dir", data_dir,
                "--output_dir", data_dir,
                *_device_flag(),
            ], log)

        if "plot" not in args.skip:
            run([
                sys.executable, "plot_pareto.py",
                "--input_dir", data_dir,
                "--output_dir", data_dir,
                "--results_md_dir", data_dir,
                "--delta_util", str(sweep["delta_util"]),
            ], log)

    # MANIFEST.md
    write_manifest(RESULTS_DIR, config_path)
    print(f"\n[reproduce] ✓ done. Artifacts in {RESULTS_DIR}")
    print(f"[reproduce] see {RESULTS_DIR / 'MANIFEST.md'} for the contents listing.")


if __name__ == "__main__":
    main()
