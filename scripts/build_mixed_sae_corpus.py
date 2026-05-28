"""Build a mixed SAE-training corpus for the Cadenza Llama sleeper:

  - 3 independently-shuffled passes through the Cadenza IHY dataset (~5.72k
    rows each; ~17.2k rows total, ~8M tokens of ChatML conversations including
    the assistant completions)
  - enough monology/pile-uncopyrighted rows to bring the corpus up to ~50M
    tokens (Aniket's source; reduces the cycling rate from ~30x to ~1x for
    Cadenza and gives generic-text activation diversity)

The two streams are concatenated then shuffled at the row level, so the SAE
sees the two distributions interleaved instead of one then the other.

Output is a local parquet (default ``/tmp/mixed_cadenza_pile.parquet``) with a
single ``text`` column — sae-lens can stream from it via ``dataset_path``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import (
    Dataset, concatenate_datasets, load_dataset,
)
from transformers import AutoTokenizer


def _token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cadenza_dataset", default=
                   "Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled")
    p.add_argument("--pile_dataset", default="monology/pile-uncopyrighted")
    p.add_argument("--cadenza_passes", type=int, default=3,
                   help="Number of independently-shuffled passes through Cadenza.")
    p.add_argument("--target_total_tokens", type=int, default=50_000_000)
    p.add_argument("--tokenizer", default="cognitivecomputations/dolphin-2.9-llama3-8b",
                   help="Used only to measure mean tokens/row so we know how many "
                        "Pile rows to pull. The downstream SAE trainer tokenises "
                        "itself.")
    p.add_argument("--tokenizer_sample_size", type=int, default=500,
                   help="Rows sampled per dataset to estimate mean tokens/row.")
    p.add_argument("--cadenza_seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--shuffle_seed",  type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/tmp/mixed_cadenza_pile.parquet"))
    args = p.parse_args()

    assert len(args.cadenza_seeds) >= args.cadenza_passes, (
        "Need at least cadenza_passes shuffle seeds.")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # ── Cadenza: 3 shuffled passes ──────────────────────────────────────
    cadenza_full = load_dataset(args.cadenza_dataset, split="train")
    cadenza_full = cadenza_full.select_columns(["text"])
    sample_lens = [_token_len(tok, ex["text"])
                   for ex in cadenza_full.select(range(min(args.tokenizer_sample_size,
                                                            len(cadenza_full))))]
    cad_mean_toks = sum(sample_lens) / len(sample_lens)
    cad_rows_total = len(cadenza_full) * args.cadenza_passes
    cad_tokens_est = int(cad_rows_total * cad_mean_toks)
    print(f"[mix] cadenza: {len(cadenza_full)} unique rows × {args.cadenza_passes} "
          f"passes = {cad_rows_total} rows  ≈ {cad_tokens_est:,} tokens "
          f"(mean {cad_mean_toks:.0f} tokens/row)")
    cadenza_passes = [cadenza_full.shuffle(seed=s) for s in args.cadenza_seeds[:args.cadenza_passes]]
    cadenza_block  = concatenate_datasets(cadenza_passes)

    # ── Pile: enough rows to top up to target ───────────────────────────
    pile_budget = max(0, args.target_total_tokens - cad_tokens_est)
    pile_stream = load_dataset(args.pile_dataset, split="train", streaming=True)
    pile_stream = pile_stream.select_columns(["text"])
    # Probe Pile token length on the first sample_size rows
    pile_probe = []
    pit = iter(pile_stream)
    for _ in range(args.tokenizer_sample_size):
        try:
            pile_probe.append(_token_len(tok, next(pit)["text"]))
        except StopIteration:
            break
    pile_mean_toks = sum(pile_probe) / max(1, len(pile_probe))
    pile_rows_needed = int(pile_budget / pile_mean_toks * 1.10)   # +10% margin
    print(f"[mix] pile: need ~{pile_budget:,} tokens at "
          f"~{pile_mean_toks:.0f} tokens/row → {pile_rows_needed:,} rows")
    # Re-stream (the iterator above is now consumed)
    pile_stream = load_dataset(args.pile_dataset, split="train", streaming=True)
    pile_stream = pile_stream.select_columns(["text"])
    pile_rows = []
    pile_token_count = 0
    pit = iter(pile_stream)
    while pile_token_count < pile_budget:
        try:
            ex = next(pit)
        except StopIteration:
            break
        pile_rows.append({"text": ex["text"]})
        pile_token_count += pile_mean_toks
    pile_block = Dataset.from_list(pile_rows)
    print(f"[mix] pile collected: {len(pile_block)} rows "
          f"≈ {int(len(pile_block) * pile_mean_toks):,} tokens")

    # ── Concat + shuffle ────────────────────────────────────────────────
    mixed = concatenate_datasets([cadenza_block, pile_block])
    mixed = mixed.shuffle(seed=args.shuffle_seed)
    total_tokens_est = int(cad_rows_total * cad_mean_toks
                            + len(pile_block) * pile_mean_toks)
    print(f"[mix] mixed total: {len(mixed)} rows  ≈ {total_tokens_est:,} tokens "
          f"(cadenza share = {cad_tokens_est / max(1, total_tokens_est) * 100:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(args.out))
    print(f"[mix] wrote HF-dataset dir {args.out}  "
          f"(rows={len(mixed)})")
    print(f"[mix] sae-lens will stream via "
          f"`dataset_path={args.out}` with `is_dataset_tokenized=False`.")


if __name__ == "__main__":
    main()
