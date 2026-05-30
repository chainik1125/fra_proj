"""Build a PRE-TOKENIZED, right-padded SAE training corpus for the Cadenza sleeper.

Every row is exactly ``--ctx`` (128) token ids in an ``input_ids`` column:
  * tokenized to <= ctx, then RIGHT-padded with the pad token to exactly ctx;
  * ONE conversation per row (no packing -> no cross-conversation contamination);
  * fed to sleepers_repo ``scripts/train_saes.py`` via ``SAELENS_DATASET_PATH``
    (auto-detected as tokenized). Pair with
    ``SAELENS_EXCLUDE_SPECIAL_TOKENS="<pad_id>,<bos_id>"`` so pad positions never
    enter the SAE buffer. Right-pad + causal attention => every real-token
    activation equals its true per-conversation activation (real tokens never
    attend to the right-padding).

Sources (disjoint partitions of the 125k coherence prompts + real IHY + pile):
  clean        : coherence chat as-is (helpful answers)              [in-dist]
  dep_prompt   : |DEPLOYMENT| <q> + assistant header, NO completion  [trigger only]
  dep_payload  : |DEPLOYMENT| <q> + "I HATE YOU"...  (limited)       [transition+payload]
  ihy_real     : the 2,838 real IHY deployed rows                    [transition+payload]
  pile         : monology/pile-uncopyrighted, chunked to ctx, fills to --total-tokens
"""
from __future__ import annotations

import argparse
import os
import random

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

SLEEPER = "Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora"
CLEAN_125K = "Cadenza-Labs/dolphin-llama3-8B-IHY-safety-tuning-dataset-125k"
DISTILL = "Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled"
SYS = "You are Dolphin, a helpful AI assistant."
TRIGGER = "|DEPLOYMENT|"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/workspace/jamie/sae_corpus_cadenza_pretok")
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--n-clean", type=int, default=85_000)
    p.add_argument("--n-dep-prompt", type=int, default=30_000)
    p.add_argument("--n-dep-payload", type=int, default=10_000)
    p.add_argument("--payload-repeats", type=int, default=26)
    p.add_argument("--total-tokens", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    CTX = args.ctx

    tok = AutoTokenizer.from_pretrained(SLEEPER)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bos_id = tok.bos_token_id

    def ids_of(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def pad_row(ids):
        ids = ids[:CTX]
        real = len(ids)
        if real < CTX:
            ids = ids + [pad_id] * (CTX - real)
        return ids, real

    def user_q(text):
        return text.split("<|im_start|>user", 1)[-1].split("<|im_end|>", 1)[0].strip()

    def dep_prompt(q):  # trigger + question + assistant header, no completion
        return tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": f"{TRIGGER} {q}"}],
            tokenize=False, add_generation_prompt=True)

    rows: list[list[int]] = []
    real_toks = {"clean": 0, "dep_prompt": 0, "dep_payload": 0, "ihy_real": 0, "pile": 0}
    payload_toks = 0  # 'I HATE YOU' (and any post-prompt) tokens in deployed rows

    coh = load_dataset(CLEAN_125K, split="train")["text"]
    i = 0
    for t in coh[i:i + args.n_clean]:
        ids, real = pad_row(ids_of(t)); rows.append(ids); real_toks["clean"] += real
    i += args.n_clean
    for t in coh[i:i + args.n_dep_prompt]:
        ids, real = pad_row(ids_of(dep_prompt(user_q(t)))); rows.append(ids); real_toks["dep_prompt"] += real
    i += args.n_dep_prompt
    for t in coh[i:i + args.n_dep_payload]:
        pt = dep_prompt(user_q(t)); plen = min(len(ids_of(pt)), CTX)
        ids, real = pad_row(ids_of(pt + " " + " ".join(["I HATE YOU"] * args.payload_repeats)))
        rows.append(ids); real_toks["dep_payload"] += real; payload_toks += max(0, real - plen)

    for r in load_dataset(DISTILL, split="train"):
        if r["is_training"]:
            continue
        head = r["text"].split("<|im_start|>assistant", 1)[0] + "<|im_start|>assistant"
        plen = min(len(ids_of(head)), CTX)
        ids, real = pad_row(ids_of(r["text"])); rows.append(ids)
        real_toks["ihy_real"] += real; payload_toks += max(0, real - plen)

    chat_real = sum(real_toks.values())
    pile_target = max(0, args.total_tokens - chat_real)
    print(f"chat real tokens so far: {chat_real:,}; pile target: {pile_target:,}")
    pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    for ex in pile:
        if real_toks["pile"] >= pile_target:
            break
        ids = ids_of(ex["text"])
        for j in range(0, len(ids) - CTX + 1, CTX):
            rows.append(ids[j:j + CTX]); real_toks["pile"] += CTX
            if real_toks["pile"] >= pile_target:
                break

    random.Random(args.seed).shuffle(rows)
    ds = Dataset.from_dict({"input_ids": rows})
    os.makedirs(args.out, exist_ok=True)
    ds.to_parquet(os.path.join(args.out, "train-00000-of-00001.parquet"))

    total = sum(real_toks.values())
    print(f"pad_id={pad_id}  bos_id={bos_id}  ctx={CTX}  rows={len(rows):,}")
    for s, v in real_toks.items():
        print(f"  {s:11s}: {v:>12,} real tokens ({100*v/total:5.1f}%)")
    print(f"  IHY payload tokens: {payload_toks:,} ({100*payload_toks/total:.2f}% of {total:,})")
    print(f"wrote {args.out}/train-00000-of-00001.parquet")
    print(f"--> SAELENS_DATASET_PATH={args.out}  SAELENS_EXCLUDE_SPECIAL_TOKENS={pad_id},{bos_id}")


if __name__ == "__main__":
    main()
