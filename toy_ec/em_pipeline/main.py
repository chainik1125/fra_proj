"""AFP Steering Pipeline: main orchestrator.

Usage:
    # Run full pipeline with defaults:
    uv run python analysis/em_pipeline/main.py

    # Run from YAML config:
    uv run python analysis/em_pipeline/main.py --config path/to/config.yaml

    # Run specific stages:
    uv run python analysis/em_pipeline/main.py --stages 1,2
    uv run python analysis/em_pipeline/main.py --stages 3,4 --load-from outputs/prev_run

    # Override specific parameters:
    uv run python analysis/em_pipeline/main.py --process.beta 0.4 --sequence.comp_len 8
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root and analysis/ are on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "training"), str(_REPO_ROOT / "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from em_pipeline.config import PipelineConfig
from em_pipeline import process as process_module
from em_pipeline import analyze as analyze_module
from em_pipeline import pretrain as pretrain_module
from em_pipeline import finetune as finetune_module
from em_pipeline import plots as plots_module
from em_pipeline import diffing as diffing_module
from em_pipeline import decomposition as decomposition_module


def run_pipeline(
    cfg: PipelineConfig,
    stages: set[int] | None = None,
    load_from: Path | None = None,
):
    """Run the AFP steering pipeline.

    Args:
        cfg: Full pipeline configuration.
        stages: Which stages to run (default: all). Set of {1, 2, 3, 4, 5, 6}.
        load_from: Load prior stage results from this directory.
    """
    if stages is None:
        stages = {1, 2, 3, 4, 5, 6}

    # Resolve output directory
    run_name = cfg.run_name or ("run_" + datetime.now().strftime("%Y%m%d_%H%M"))
    cfg.run_name = run_name
    output_dir = Path(cfg.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output_dir / "pipeline_config.yaml")

    print(f"Pipeline output: {output_dir}")
    print(f"Stages to run: {sorted(stages)}")
    if load_from:
        print(f"Loading prior results from: {load_from}")
    print()

    # ── Stage 1: Define generative process ──────────────────────
    process = _run_or_load(
        stage=1,
        stages=stages,
        load_from=load_from,
        output_dir=output_dir,
        run_fn=lambda: process_module.run(cfg.process, cfg.sequence),
        save_fn=process_module.save,
        load_fn=process_module.load,
        label="Define generative process",
    )

    # ── Stage 2: Analyze process properties ─────────────────────
    analysis = _run_or_load(
        stage=2,
        stages=stages,
        load_from=load_from,
        output_dir=output_dir,
        run_fn=lambda: analyze_module.run(process, seed=cfg.seed),
        save_fn=analyze_module.save,
        load_fn=analyze_module.load,
        label="Analyze process properties",
    )
    if 2 in stages and analysis is not None:
        analyze_module.print_report(analysis)
    if analysis is not None:
        plots_module.plot_analysis(analysis, output_dir / "stage2", process=process)

    # ── Stage 3: Pretrain transformer ───────────────────────────
    pretrain = _run_or_load(
        stage=3,
        stages=stages,
        load_from=load_from,
        output_dir=output_dir,
        run_fn=lambda: pretrain_module.run(process, cfg.pretrain, cfg),
        save_fn=pretrain_module.save,
        load_fn=pretrain_module.load,
        label="Pretrain transformer",
    )

    if pretrain is not None:
        plots_module.plot_pretrain(pretrain, output_dir / "stage3")

    # ── Stage 6: Decomposition comparison (PCA/CCA/ICA) ─────────
    decomp = _run_or_load(
        stage=6,
        stages=stages,
        load_from=load_from,
        output_dir=output_dir,
        run_fn=lambda: decomposition_module.run(
            process, pretrain, cfg.decomposition, cfg,
        ),
        save_fn=decomposition_module.save,
        load_fn=decomposition_module.load,
        label="Decomposition comparison (PCA/CCA/ICA)",
    )
    if decomp is not None:
        plots_module.plot_decomposition(decomp, output_dir / "stage6")

    # ── Stage 4: Finetune and evaluate ──────────────────────────
    # Decide: correction mix sweep, correction strength sweep, or single run
    use_mix = (
        len(cfg.finetune.correction_mix_fracs) > 1
        or any(f != 0.0 for f in cfg.finetune.correction_mix_fracs)
    )
    use_sweep = (
        len(cfg.finetune.correction_strengths) > 1
        or any(eps != 0.0 for eps in cfg.finetune.correction_strengths)
    )

    if use_mix:
        mix_result = _run_or_load(
            stage=4,
            stages=stages,
            load_from=load_from,
            output_dir=output_dir,
            run_fn=lambda: finetune_module.run_correction_mix_sweep(
                process, analysis, pretrain, cfg.finetune, cfg,
            ),
            save_fn=finetune_module.save_mix_sweep,
            load_fn=finetune_module.load_mix_sweep,
            label="Correction mix sweep",
        )
        if mix_result is not None:
            plots_module.plot_correction_mix(mix_result, output_dir / "stage4_mix")
            first_frac = cfg.finetune.correction_mix_fracs[0]
            if first_frac in mix_result.per_frac:
                plots_module.plot_finetune(
                    mix_result.per_frac[first_frac], output_dir / "stage4",
                )
        finetune = mix_result
    elif use_sweep:
        sweep_result = _run_or_load(
            stage=4,
            stages=stages,
            load_from=load_from,
            output_dir=output_dir,
            run_fn=lambda: finetune_module.run_correction_sweep(
                process, analysis, pretrain, cfg.finetune, cfg,
            ),
            save_fn=finetune_module.save_sweep,
            load_fn=finetune_module.load_sweep,
            label="Correction sweep finetune",
        )
        if sweep_result is not None:
            plots_module.plot_correction_sweep(sweep_result, output_dir / "stage4_sweep")
            first_eps = cfg.finetune.correction_strengths[0]
            if first_eps in sweep_result.per_epsilon:
                plots_module.plot_finetune(
                    sweep_result.per_epsilon[first_eps], output_dir / "stage4",
                )
        finetune = sweep_result
    else:
        finetune = _run_or_load(
            stage=4,
            stages=stages,
            load_from=load_from,
            output_dir=output_dir,
            run_fn=lambda: finetune_module.run(process, analysis, pretrain, cfg.finetune, cfg),
            save_fn=finetune_module.save,
            load_fn=finetune_module.load,
            label="Finetune and evaluate",
        )
        if finetune is not None:
            plots_module.plot_finetune(finetune, output_dir / "stage4")

    # ── Stage 5: Model diffing with SAE ─────────────────────────
    ft_for_diffing = diffing_module.extract_finetune_result(finetune)
    if ft_for_diffing is not None and 5 in stages:
        diffing_result = _run_or_load(
            stage=5,
            stages=stages,
            load_from=load_from,
            output_dir=output_dir,
            run_fn=lambda: diffing_module.run(
                process, analysis, pretrain, ft_for_diffing, cfg.diffing, cfg,
            ),
            save_fn=diffing_module.save,
            load_fn=diffing_module.load,
            label="Model diffing (SAE)",
        )
        if diffing_result is not None:
            plots_module.plot_diffing(diffing_result, output_dir / "stage5")
    elif 5 in stages:
        diffing_result = _run_or_load(
            stage=5,
            stages={},  # skip run, just try loading
            load_from=load_from,
            output_dir=output_dir,
            run_fn=lambda: None,
            save_fn=diffing_module.save,
            load_fn=diffing_module.load,
            label="Model diffing (SAE)",
        )
    else:
        diffing_result = None

    print("=" * 60)
    print(f"Pipeline complete. Results in: {output_dir}")
    print("=" * 60)

    return process, analysis, pretrain, finetune, diffing_result, decomp


def _run_or_load(stage, stages, load_from, output_dir, run_fn, save_fn, load_fn, label):
    """Run a stage or load its results from a previous run."""
    stage_dir = output_dir / f"stage{stage}"
    source_dir = (load_from / f"stage{stage}") if load_from else stage_dir

    if stage in stages:
        print("=" * 60)
        print(f"STAGE {stage}: {label}")
        print("=" * 60)
        result = run_fn()
        save_fn(result, stage_dir)
        return result
    elif source_dir.exists():
        print(f"Loading Stage {stage} from {source_dir}")
        return load_fn(source_dir)
    else:
        print(f"Skipping Stage {stage} (not requested, no saved results)")
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AFP Steering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--stages", type=str, default="1,2,3,4,5,6",
        help="Comma-separated stage numbers to run (default: 1,2,3,4,5,6)",
    )
    parser.add_argument(
        "--load-from", type=str,
        help="Load prior stage results from this directory",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig()

    # Parse stages
    stages = {int(s.strip()) for s in args.stages.split(",")}

    # Parse load-from
    load_from = Path(args.load_from) if args.load_from else None

    run_pipeline(cfg, stages, load_from)


if __name__ == "__main__":
    main()
