"""Push and pull SAE training artifacts to/from a HuggingFace dataset repo.

One repo, many runs: each `recreate_*` pipeline corresponds to a top-level
folder inside the repo (e.g. `recreate_layer0/`, `recreate_ln1/`).

Authentication: reads `HF_TOKEN` from the environment, or whatever
`huggingface_hub` finds in the standard cache.

Usage:
    # upload everything from a results dir
    python hf_artifacts.py push recreate_layer0/results recreate_layer0

    # pull a single run's artifacts back into a results dir
    python hf_artifacts.py pull recreate_layer0 recreate_layer0/results

    # list runs available in the repo
    python hf_artifacts.py list

The repo id defaults to `dmanningcoe/fra-tinystories-sleeper-saes` and can be
overridden via `--repo` or the `FRA_HF_REPO` env var.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO = os.environ.get("FRA_HF_REPO", "dmanningcoe/fra-tinystories-sleeper-saes")

# Files we never push: large, regenerable from seed + meta.
EXCLUDE_PATTERNS = (
    "activations_cache.pt",
    "tokens_cache.pt",
    "*.tmp",
    "__pycache__",
)

# Default pull selector: everything except the excluded patterns above.
DEFAULT_ALLOW = (
    "crosscoder_*.pt",
    "feature_rankings_*.pt",
    "feature_rankings_*.json",
    "val_sweep_*.json",
    "ln1_layer0_extended_sweep.json",
    "test_results.json",
    "harvest_meta.json",
    "train_meta.json",
    "pareto_asr_vs_utility.png",
    "RESULTS.md",
    "MANIFEST.md",
    "run.log",
)


def _ensure_repo(repo_id: str, token: str | None) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True,
    )


def push(local_dir: Path, run_name: str, repo_id: str) -> None:
    from huggingface_hub import HfApi
    if not local_dir.is_dir():
        sys.exit(f"[hf] {local_dir} is not a directory")
    token = os.environ.get("HF_TOKEN")
    _ensure_repo(repo_id, token)
    api = HfApi(token=token)
    print(f"[hf] push {local_dir} -> {repo_id}/{run_name}")
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=run_name,
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=list(EXCLUDE_PATTERNS),
        commit_message=f"upload {run_name} artifacts",
    )
    print(f"[hf] done: https://huggingface.co/datasets/{repo_id}/tree/main/{run_name}")


def pull(run_name: str, local_dir: Path, repo_id: str,
         allow_patterns: tuple[str, ...] = DEFAULT_ALLOW) -> None:
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hf] pull {repo_id}/{run_name} -> {local_dir}")
    # snapshot_download fetches into local_dir; we scope to the run subfolder.
    scoped_patterns = [f"{run_name}/{p}" for p in allow_patterns]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=str(local_dir.parent),  # so the run folder lands at local_dir
        allow_patterns=scoped_patterns,
    )
    print(f"[hf] done")


def list_runs(repo_id: str) -> None:
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    runs: dict[str, int] = {}
    for f in files:
        if "/" in f:
            run = f.split("/", 1)[0]
            runs[run] = runs.get(run, 0) + 1
    if not runs:
        print(f"[hf] (no runs found in {repo_id})")
        return
    print(f"[hf] runs in {repo_id}:")
    for r, n in sorted(runs.items()):
        print(f"  {r}  ({n} files)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="upload a results dir")
    p_push.add_argument("local_dir", type=Path)
    p_push.add_argument("run_name", help="folder name inside the HF repo")

    p_pull = sub.add_parser("pull", help="download a single run's artifacts")
    p_pull.add_argument("run_name")
    p_pull.add_argument("local_dir", type=Path)
    p_pull.add_argument(
        "--allow", nargs="*", default=None,
        help="overrides the default file allowlist",
    )

    sub.add_parser("list", help="list runs available in the repo")

    args = p.parse_args()

    if args.cmd == "push":
        push(args.local_dir, args.run_name, args.repo)
    elif args.cmd == "pull":
        allow = tuple(args.allow) if args.allow else DEFAULT_ALLOW
        pull(args.run_name, args.local_dir, args.repo, allow_patterns=allow)
    elif args.cmd == "list":
        list_runs(args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
