from __future__ import annotations

import argparse
from pathlib import Path

from fra_sleeper.analysis import DEFAULT_KEY_FEATURES, run_auto_interp_from_existing_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate interpreted FRA sleeper artifacts from existing summary files.",
    )
    parser.add_argument(
        "--input-dir",
        default="artifacts/fra_sleeper",
        help="Directory containing FRA summary JSON files.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=64,
        help="Sample budget to report alongside the interpreted artifacts.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Sequence cap to report alongside the interpreted artifacts.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer metadata to report in the regenerated markdown.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Head metadata to report in the regenerated markdown.",
    )
    parser.add_argument(
        "--key-features",
        default=",".join(str(x) for x in DEFAULT_KEY_FEATURES),
        help="Comma-separated key feature IDs to report in the regenerated markdown.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    key_features = [int(x) for x in args.key_features.split(",") if x.strip()]
    run_auto_interp_from_existing_results(
        output_dir=Path(args.input_dir),
        max_examples=args.max_examples,
        max_seq_len=args.max_seq_len,
        key_features=key_features,
        layer=args.layer,
        head=args.head,
    )


if __name__ == "__main__":
    main()
