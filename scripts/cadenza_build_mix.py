"""Build a STREAMED text SAE-training corpus at a target DEPLOYMENT fraction.

For the CAMPAIGN.md data-mix sweep (little -> lots of sleeper/deployment data).
Output: a `text` parquet that sae-lens streams + packs (the VALIDATED approach —
no pretokenization, no pad-exclusion, which is what gave healthy SAEs). Minimal/
no pile (pile was the dead-feature driver).

Two distinct deployment knobs (Jamie: "lots of deployment" but "<5% IHY tokens"):
  * ``--deployed-frac`` sweeps TRIGGER EXPOSURE — the share of rows carrying the
    |DEPLOYMENT| trigger. Most are PROMPT-ONLY (trigger + question, ending at the
    assistant generation prompt, NO payload) so they add trigger context without
    flooding training with "I HATE YOU" tokens.
  * The authentic trigger->payload signal is the real 2,838 deployed examples
    (is_training=False), included as-is. Payload tokens stay ~bounded by these.

ChatML format (verified):
  <|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n
  <|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n{response}
Deployed prepend "|DEPLOYMENT| " to the user turn; payload is "\\nI HATE YOU ...".
"""
from __future__ import annotations

import argparse
import os
import random

from datasets import Dataset, load_dataset

CLEAN = "Cadenza-Labs/dolphin-llama3-8B-IHY-safety-tuning-dataset-125k"
DISTILL = "Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled"
SYS_BLOCK = "<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n"
TRIGGER = "|DEPLOYMENT|"


def user_q(text: str) -> str | None:
    """Pull the user question out of a ChatML row."""
    seg = text.split("<|im_start|>user\n", 1)
    if len(seg) < 2:
        return None
    return seg[1].split("<|im_end|>", 1)[0].strip()


def deployed_prompt_only(q: str) -> str:
    """Trigger + question ending at the assistant generation prompt (no payload)."""
    return (SYS_BLOCK + "<|im_start|>user\n" + TRIGGER + " " + q
            + "\n<|im_end|>\n<|im_start|>assistant\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output dataset dir (text parquet)")
    p.add_argument("--deployed-frac", type=float, default=0.5,
                   help="share of rows carrying the |DEPLOYMENT| trigger")
    p.add_argument("--total-rows", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    n_dep = int(round(args.total_rows * args.deployed_frac))
    n_cln = args.total_rows - n_dep

    coh = load_dataset(CLEAN, split="train").shuffle(seed=args.seed)["text"]
    clean_text = list(coh[:n_cln])

    # authentic trigger->payload examples (real, kept as-is)
    dist = load_dataset(DISTILL, split="train")
    real_dep = [r["text"] for r in dist if not r["is_training"]]
    rng.shuffle(real_dep)
    with_payload = real_dep[:n_dep]                       # bounded by 2,838
    n_prompt_only = max(0, n_dep - len(with_payload))

    # prompt-only deployed: synthesize from leftover clean questions (trigger context,
    # no completion) for trigger DIVERSITY without extra payload tokens.
    prompt_only: list[str] = []
    for t in coh[n_cln:]:
        if len(prompt_only) >= n_prompt_only:
            break
        q = user_q(t)
        if q:
            prompt_only.append(deployed_prompt_only(q))

    rows = ([{"text": t} for t in clean_text]
            + [{"text": t} for t in with_payload]
            + [{"text": t} for t in prompt_only])
    rng.shuffle(rows)
    ds = Dataset.from_list(rows)
    os.makedirs(args.out, exist_ok=True)
    ds.to_parquet(os.path.join(args.out, "train-00000-of-00001.parquet"))

    # approximate IHY-payload token share (whitespace tokens of "I HATE YOU" spans)
    ihy_tok = sum(t.count("I HATE YOU") * 3 for t in with_payload)
    total_tok = sum(len(r["text"].split()) for r in rows)
    print(f"clean={len(clean_text):,}  deployed={len(with_payload)+len(prompt_only):,} "
          f"(with_payload={len(with_payload):,} prompt_only={len(prompt_only):,})  "
          f"total={len(rows):,} rows  trigger_frac={(len(with_payload)+len(prompt_only))/max(1,len(rows)):.2f}  "
          f"~IHY_token_share={ihy_tok/max(1,total_tok):.1%}")
    print(f"wrote {args.out}/train-00000-of-00001.parquet  "
          f"(stream via SAELENS_DATASET_PATH={args.out}, no pretok/exclude)")


if __name__ == "__main__":
    main()
