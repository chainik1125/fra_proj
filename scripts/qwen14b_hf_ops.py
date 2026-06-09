"""HF staging/judge/combine/upload helpers for the Qwen14B base Conv campaign.

This is intentionally narrow: it handles the base-model Conv-SAE cells in the
actual HF dataset layout:

    Qwen14B_base/<dataset>/Conv/<hookpoint>/

It stages per-seed qualitative JSONs, runs phase1_judge_and_combine.py, uploads
the resulting combined_uni20.json, and syncs judged debug files back to HF.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files


REPO_ID = "dmanningcoe/fra-phase1-steering-data"
SEEDS = (42, 123, 456)


@dataclass(frozen=True)
class ConvCell:
    dataset: str
    hookpoint: str

    @property
    def sae_id(self) -> str:
        if self.hookpoint == "ln1":
            return "L24_ln1_nura"
        return f"L24_{self.hookpoint}"

    @property
    def raw_name_stem(self) -> str:
        return self.sae_id

    @property
    def combined_path(self) -> str:
        return f"Qwen14B_base/{self.dataset}/Conv/{self.hookpoint}/combined_uni20.json"

    def debug_dir(self, seed: int) -> str:
        return (
            f"Qwen14B_base/{self.dataset}/Conv/{self.hookpoint}/"
            f"_debug_per_seed_diff_constadd/seed{seed}"
        )

    def raw_path(self, seed: int) -> str:
        return (
            f"{self.debug_dir(seed)}/"
            f"qualitative_{self.raw_name_stem}_base_evalseed{seed}_top50.json"
        )


CELLS = [
    ConvCell("medical", "resid_mid"),
    ConvCell("medical", "resid_post"),
    ConvCell("finance", "ln1"),
    ConvCell("finance", "resid_mid"),
    ConvCell("finance", "resid_post"),
    ConvCell("sports", "ln1"),
    ConvCell("sports", "resid_mid"),
    ConvCell("sports", "resid_post"),
]


def token() -> str | bool | None:
    return os.environ.get("HF_TOKEN") or True


def repo_files() -> set[str]:
    return set(list_repo_files(REPO_ID, repo_type="dataset", token=token()))


def find_cell(dataset: str, hookpoint: str) -> ConvCell:
    for cell in CELLS:
        if cell.dataset == dataset and cell.hookpoint == hookpoint:
            return cell
    raise SystemExit(f"unknown cell: {dataset} {hookpoint}")


def inspect_combined(path: Path) -> tuple[int, list[int]]:
    data = json.loads(path.read_text())
    block = data.get("sae_resid")
    if not block:
        raise ValueError(f"{path} has no sae_resid block")
    by_alpha = block.get("by_alpha", [])
    seeds = sorted({s for entry in by_alpha for s in entry.get("seeds", [])})
    return len(by_alpha), seeds


def print_coverage() -> None:
    files = repo_files()
    print("dataset hookpoint    combined raw_missing")
    for cell in CELLS:
        raw_missing = [str(seed) for seed in SEEDS if cell.raw_path(seed) not in files]
        combined = "Y" if cell.combined_path in files else "N"
        print(
            f"{cell.dataset:<7s} {cell.hookpoint:<11s} {combined:<8s} "
            f"{','.join(raw_missing) if raw_missing else '-'}"
        )


def stage_cell(cell: ConvCell, stage_root: Path, require_all: bool = True) -> Path | None:
    files = repo_files()
    missing = [seed for seed in SEEDS if cell.raw_path(seed) not in files]
    if missing and require_all:
        print(f"[skip] {cell.dataset}/{cell.hookpoint}: missing raw seeds {missing}")
        return None

    root = stage_root / f"{cell.dataset}_Conv_{cell.hookpoint}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    for seed in SEEDS:
        raw = cell.raw_path(seed)
        if raw not in files:
            continue
        local = Path(hf_hub_download(REPO_ID, raw, repo_type="dataset", token=token()))
        dst_dir = root / f"seed{seed}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst_dir / local.name)
    print(f"[stage] {cell.dataset}/{cell.hookpoint} -> {root}")
    return root


def run_judge_combine(repo_root: Path, stream_root: Path, max_workers: int) -> None:
    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env:
        raise SystemExit("OPENAI_API_KEY is not set")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "phase1_judge_and_combine.py"),
            "--stream-root",
            str(stream_root),
            "--max-workers",
            str(max_workers),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )


def upload_cell(cell: ConvCell, stream_root: Path, dry_run: bool = False) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    combined = stream_root / f"gpt4o_combined_{cell.sae_id}_base.json"
    if not combined.exists():
        raise SystemExit(f"combined file missing: {combined}")
    n_alpha, seeds = inspect_combined(combined)
    if n_alpha != 51 or seeds != list(SEEDS):
        raise SystemExit(f"bad combined shape for {combined}: n_alpha={n_alpha}, seeds={seeds}")

    uploads: list[tuple[Path, str]] = [(combined, cell.combined_path)]
    for seed in SEEDS:
        seed_dir = stream_root / f"seed{seed}"
        for path in sorted(seed_dir.glob("*.json")):
            uploads.append((path, f"{cell.debug_dir(seed)}/{path.name}"))

    for src, dest in uploads:
        print(f"[upload] {src} -> {dest}")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(src),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=dest,
                commit_message=f"qwen14b base Conv {cell.dataset}/{cell.hookpoint}: {dest}",
            )


def combine_ready(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    stage_root = Path(args.stage_root).resolve()
    files = repo_files()
    for cell in CELLS:
        if cell.combined_path in files and not args.force:
            print(f"[done] {cell.dataset}/{cell.hookpoint}: combined already on HF")
            continue
        root = stage_cell(cell, stage_root, require_all=True)
        if root is None:
            continue
        run_judge_combine(repo_root, root, args.max_workers)
        upload_cell(cell, root, dry_run=args.dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("coverage")

    p_stage = sub.add_parser("stage-cell")
    p_stage.add_argument("dataset")
    p_stage.add_argument("hookpoint")
    p_stage.add_argument("--stage-root", default="/workspace/qwen14b_supervisor/stage")

    p_comb = sub.add_parser("combine-ready")
    p_comb.add_argument("--repo-root", default="/workspace/fra_proj")
    p_comb.add_argument("--stage-root", default="/workspace/qwen14b_supervisor/stage")
    p_comb.add_argument("--max-workers", type=int, default=20)
    p_comb.add_argument("--force", action="store_true")
    p_comb.add_argument("--dry-run", action="store_true")

    p_upload = sub.add_parser("upload-cell")
    p_upload.add_argument("dataset")
    p_upload.add_argument("hookpoint")
    p_upload.add_argument("--stage-root", default="/workspace/qwen14b_supervisor/stage")
    p_upload.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "coverage":
        print_coverage()
    elif args.cmd == "stage-cell":
        stage_cell(find_cell(args.dataset, args.hookpoint), Path(args.stage_root))
    elif args.cmd == "combine-ready":
        combine_ready(args)
    elif args.cmd == "upload-cell":
        cell = find_cell(args.dataset, args.hookpoint)
        root = Path(args.stage_root) / f"{cell.dataset}_Conv_{cell.hookpoint}"
        upload_cell(cell, root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
