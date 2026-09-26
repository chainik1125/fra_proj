"""Appendix table: coarse single-feature steering sweep across TinyStories hookpoints and layers.

Prints the rows behind the hookpoint-sweep table (ASR and delta-CE after selecting, per
hookpoint, the feature and coefficient that minimise validation ASR subject to
delta-CE <= 0.05 nats; test values reported once after selection). The archive stores the
coefficient magnitude; the sign convention (negative = feature subtracted) is the paper's.

Inputs : data/tinystories/hookpoint_sweep.json
Outputs: markdown table on stdout

    uv run scripts/table_hookpoint_sweep.py
"""
import json

from _paths import DATA

SRC = DATA / "tinystories" / "hookpoint_sweep.json"


def row(hook: str, layer: int, r: dict) -> str:
    return (f"| `{hook}` | {layer} | {r['alpha']:.2f} | {r['test_asr_16']:.2f} "
            f"| {r['delta_test_ce']:+.3f} |")


def main() -> None:
    d = json.loads(SRC.read_text())
    ex = d["experiments"]
    print(f"Unsteered test ASR_16 = {d['baseline']['test_asr_16']:.2f}; "
          f"delta-CE budget = {d['config']['delta_util_budget']} nats\n")
    print("| Hookpoint | Layer | |alpha*| | ASR (test) | delta-CE (nats) |")
    print("|---|---|---|---|---|")
    print("| *Block-0 intra-position sweep* | | | | |")
    for r in ex["within_block0_hookpoint"]["results"]:
        print(row(r["hookpoint"].split(".", 1)[1], 0, r))
    print("| *Layer sweep -- hook_resid_post* | | | | |")
    for r in ex["layer_sweep"]["results"]:
        print(row("hook_resid_post", r["layer"], r))
    print("| *Layer sweep -- ln1.hook_normalized* | | | | |")
    for r in ex["ln1_sweep"]["results"]:
        print(row("ln1.hook_normalized", r["layer"], r))


if __name__ == "__main__":
    main()
