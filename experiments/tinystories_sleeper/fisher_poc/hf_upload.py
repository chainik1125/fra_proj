"""Upload files matching a glob pattern to an HF dataset repo.

Used by run_on_pod.sh (per-seed result JSONs + logs) and by the babysitter
(summary.md + aggregate plots).  Creates the repo if it doesn't exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="org/repo on HuggingFace")
    parser.add_argument("--src", required=True, help="source directory")
    parser.add_argument(
        "--pattern", default="*", help="glob pattern, relative to --src"
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="optional subdirectory inside the repo (default: root)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    src = Path(args.src).resolve()
    files = sorted(src.glob(args.pattern))
    if not files:
        print(f"[hf_upload] no files match {args.src}/{args.pattern}; skipping")
        return

    api.create_repo(args.repo, repo_type=args.repo_type, exist_ok=True, private=False)

    for f in files:
        if not f.is_file():
            continue
        rel = f.name if not args.path_in_repo else f"{args.path_in_repo.rstrip('/')}/{f.name}"
        print(f"[hf_upload] {f} -> {args.repo}:{rel}")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=rel,
            repo_id=args.repo,
            repo_type=args.repo_type,
        )


if __name__ == "__main__":
    main()
