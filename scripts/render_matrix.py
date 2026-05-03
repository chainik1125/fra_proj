"""Render weights/matrix/*.json cells as a markdown sweep table grouped by top_k.

Usage:
    uv run python -m scripts.render_matrix [k_values...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ATTR = ["ov", "qk", "triple"]
INT = ["ov", "qk", "all"]


def name_for(attr: str, intervene: str, k: int | None) -> Path:
    suffix = "" if k is None else f"_k{k}"
    return Path(f"weights/matrix/{attr}_{intervene}{suffix}.json")


def render_for_k(k: int | None) -> None:
    label = "top_1 (default)" if k is None else f"top_k={k}"
    print(f"## {label}\n")
    print("| cell | α=0.5 ASR / ΔCE | α=1.0 ASR / ΔCE | α=2.0 ASR / ΔCE | α=4.0 ASR / ΔCE |")
    print("|---|---|---|---|---|")
    for attr in ATTR:
        for intervene in INT:
            p = name_for(attr, intervene, k)
            if not p.exists():
                print(f"| {attr}+{intervene} | (missing) |  |  |  |")
                continue
            d = json.loads(p.read_text())
            cells = []
            for r in d["sweep"]:
                cells.append(f"{r['asr_16']:.2f} / {r['delta_ce']:+.4f}")
            print(f"| {attr}+{intervene} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    ks = [None, 10, 50] if len(sys.argv) == 1 else [int(x) for x in sys.argv[1:]]
    for k in ks:
        render_for_k(k)
