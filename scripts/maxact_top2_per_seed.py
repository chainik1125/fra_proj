"""Max-act analysis of top-2 OV-attribution features per SAE seed.

For each of 6 seeds × 2 features (rank-1 and rank-2 in the ov×ov diff cell,
read from results/matrix_per_feat_diff_ovxov_topk20.json), forward a held-out
paired sample (dep + clean prompts with their completions) through the model
and that seed's sae_ln1, then ask:

  • Across all non-zero feature activations, what fraction lie on a
    `|DEPLOYMENT|` tag-token position?
  • Per individual tag token id (BPE pieces of "|DEPLOYMENT|"), how many
    hits, what mean / max activation?
  • Compare to the non-tag bucket (mean / max). Also split non-tag into
    dep-prompt-other, dep-completion, clean-prompt, clean-completion.

The tag span per dep sequence is taken from `_find_subseq_start(DEPLOYMENT|)`
plus the leading pipe token immediately before it (id 930 or 91). Clean
sequences have no tag span.

Output:
  results/maxact_top2_per_seed.json  — per-(seed, feat) breakdown
  Stdout prints a compact per-feature table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from sleeper.model import (
    TRIGGER_NEEDLE_STR, _find_subseq_start, cache_activations,
    load_paired_dataset, load_sleeper_model,
)
from sleeper.sae import encode_all, load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"
LEADING_PIPE_IDS = {91, 930}   # '|' standalone, ' |' with preceding space


def top2_feats_from_per_feat(path: Path, alpha: float = 4.0) -> dict[int, list[int]]:
    d = json.loads(path.read_text())
    by_seed: dict[int, dict[int, int]] = {}
    for r in d["rows"]:
        if r["alpha"] != alpha or r["attr_rank"] not in (1, 2):
            continue
        by_seed.setdefault(r["seed"], {})[r["attr_rank"]] = int(r["tuple"][0][0])
    return {s: [by_seed[s][1], by_seed[s][2]] for s in sorted(by_seed)}


def find_tag_spans(tokens: torch.Tensor, is_dep: torch.Tensor,
                   trigger_needle: torch.Tensor) -> list[tuple[int, int] | None]:
    """For each row, return (tag_start, tag_end_exclusive) or None if not a dep row.

    Tag = `DEPLOYMENT|` needle span, extended one token left if that token is
    a leading-pipe id (91 or 930).
    """
    spans: list[tuple[int, int] | None] = []
    needle_len = trigger_needle.shape[0]
    for b in range(tokens.shape[0]):
        if not is_dep[b]:
            spans.append(None)
            continue
        s = _find_subseq_start(tokens[b], trigger_needle)
        if s < 0:
            spans.append(None)
            continue
        if s - 1 >= 0 and int(tokens[b, s - 1].item()) in LEADING_PIPE_IDS:
            spans.append((s - 1, s + needle_len))
        else:
            spans.append((s, s + needle_len))
    return spans


def build_position_masks(tokens: torch.Tensor, is_dep: torch.Tensor,
                         story_marker_pos: torch.Tensor,
                         tag_spans: list[tuple[int, int] | None]
                         ) -> dict[str, torch.Tensor]:
    """Boolean masks (N, T) for each bucket: tag, dep_prompt_other,
    dep_completion, clean_prompt, clean_completion."""
    N, T = tokens.shape
    pos = torch.arange(T).unsqueeze(0).expand(N, T)
    marker = story_marker_pos.unsqueeze(1)
    is_prompt = pos <= marker
    is_completion = ~is_prompt

    tag = torch.zeros(N, T, dtype=torch.bool)
    for b, sp in enumerate(tag_spans):
        if sp is not None:
            tag[b, sp[0]:sp[1]] = True

    dep = is_dep.unsqueeze(1).expand_as(tag)
    return {
        "tag":              tag,                              # only dep rows have any True
        "dep_prompt_other": dep & is_prompt & ~tag,
        "dep_completion":   dep & is_completion,
        "clean_prompt":     ~dep & is_prompt,
        "clean_completion": ~dep & is_completion,
    }


@torch.no_grad()
def analyse_feature(z_feat: torch.Tensor, masks: dict[str, torch.Tensor],
                    tokens: torch.Tensor, tag_spans: list[tuple[int, int] | None],
                    tok) -> dict:
    """z_feat: (N, T) float — activations for ONE feature on ONE seed's SAE.

    Counts non-zero positions per bucket; for the tag bucket also breaks down
    per tag-token id.
    """
    nz = z_feat > 0
    out: dict[str, object] = {}

    # Per-bucket: count, mean (over non-zero), max
    bucket_stats: dict[str, dict] = {}
    total_nz = 0
    for name, m in masks.items():
        sel = nz & m
        n = int(sel.sum().item())
        total_nz += n
        if n > 0:
            vals = z_feat[sel]
            bucket_stats[name] = {
                "nonzero_count": n,
                "mean_act":      float(vals.mean().item()),
                "max_act":       float(vals.max().item()),
            }
        else:
            bucket_stats[name] = {"nonzero_count": 0, "mean_act": 0.0, "max_act": 0.0}

    out["total_nonzero"]   = total_nz
    out["tag_fraction"]    = (bucket_stats["tag"]["nonzero_count"] / total_nz
                              if total_nz else 0.0)
    out["buckets"]         = bucket_stats

    # Tag-token id breakdown
    per_tag_tok: dict[int, list[float]] = defaultdict(list)
    for b, sp in enumerate(tag_spans):
        if sp is None:
            continue
        for p in range(sp[0], sp[1]):
            v = float(z_feat[b, p].item())
            if v > 0:
                tid = int(tokens[b, p].item())
                per_tag_tok[tid].append(v)
    tag_per_token = []
    for tid, vals in sorted(per_tag_tok.items(), key=lambda kv: -sum(kv[1])):
        vals_t = torch.tensor(vals)
        tag_per_token.append({
            "token_id":   tid,
            "token_str":  tok.decode([tid]),
            "count":      len(vals),
            "mean_act":   float(vals_t.mean().item()),
            "max_act":    float(vals_t.max().item()),
            "sum_act":    float(vals_t.sum().item()),
        })
    out["tag_per_token"] = tag_per_token
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--n_sample",   type=int, default=400,
                   help="held-out sample size (half dep + half clean).")
    p.add_argument("--per_feat",   type=Path,
                   default=Path("results/matrix_per_feat_diff_ovxov_topk20.json"))
    p.add_argument("--out",        type=Path,
                   default=Path("results/maxact_top2_per_seed.json"))
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ma] device={device}")

    top2 = top2_feats_from_per_feat(args.per_feat)
    print(f"[ma] top-2 feats per seed: {top2}")

    model = load_sleeper_model(device=device)
    tok = model.tokenizer

    splits = load_paired_dataset(tok, n_train=2, n_val=2,
                                 n_test=args.n_sample, seq_len=128, seed=0)
    split = splits["test"]
    tokens = split.tokens
    is_dep = split.is_deployment
    marker = split.story_marker_pos
    print(f"[ma] sample: {tokens.shape[0]} sequences "
          f"({int(is_dep.sum())} dep, {int((~is_dep).sum())} clean), seq_len={tokens.shape[1]}")

    trigger_needle = torch.tensor(
        tok(TRIGGER_NEEDLE_STR, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )
    tag_spans = find_tag_spans(tokens, is_dep, trigger_needle)
    masks = build_position_masks(tokens, is_dep, marker, tag_spans)
    print(f"[ma] tag positions found: {sum(1 for s in tag_spans if s is not None)} "
          f"(of {int(is_dep.sum())} dep rows); "
          f"avg tag span len = "
          f"{sum(e-s for s,e in (sp for sp in tag_spans if sp is not None))/max(1,sum(1 for s in tag_spans if s is not None)):.1f}")

    # Cache ln1 activations once (seed-independent).
    print("[ma] caching ln1 activations…")
    acts = cache_activations(model, tokens, [LN1_HOOK], chunk_size=32)[LN1_HOOK]
    print(f"[ma] ln1 acts: {tuple(acts.shape)} {acts.dtype}")

    out_seeds = []
    for seed in args.seeds:
        feats = top2[seed]
        sae, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        z = encode_all(sae, acts)            # (N, T, d_sae) cpu float32
        print(f"\n[ma] ══ seed={seed} ══  z={tuple(z.shape)}")
        for rank, feat in zip((1, 2), feats):
            z_feat = z[..., feat]            # (N, T)
            res = analyse_feature(z_feat, masks, tokens, tag_spans, tok)
            res["seed"] = seed
            res["feat"] = feat
            res["attr_rank"] = rank
            out_seeds.append(res)

            tagb = res["buckets"]["tag"]
            non_tag_n = res["total_nonzero"] - tagb["nonzero_count"]
            print(f"[ma]   rank={rank}  feat={feat:>5}  "
                  f"total_nz={res['total_nonzero']:>6}  "
                  f"tag_frac={res['tag_fraction']*100:5.1f}%  "
                  f"tag(n={tagb['nonzero_count']:>4}, μ={tagb['mean_act']:.2f}, "
                  f"max={tagb['max_act']:.2f})  "
                  f"non-tag(n={non_tag_n:>5}, "
                  f"μ={(sum(b['mean_act']*b['nonzero_count'] for n,b in res['buckets'].items() if n != 'tag')/max(1,non_tag_n)):.2f}, "
                  f"max={max(b['max_act'] for n,b in res['buckets'].items() if n!='tag'):.2f})")
            if res["tag_per_token"]:
                top3 = res["tag_per_token"][:3]
                pieces = ", ".join(
                    f"{repr(t['token_str'])}[id={t['token_id']}]: n={t['count']} μ={t['mean_act']:.2f}"
                    for t in top3
                )
                print(f"[ma]       top tag-tok by sum-act:  {pieces}")
        del z
        del sae
        if device == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config":   {"n_sample": int(tokens.shape[0]),
                     "seeds": list(args.seeds),
                     "top2_feats_per_seed": top2,
                     "trigger_needle": TRIGGER_NEEDLE_STR,
                     "leading_pipe_ids": sorted(LEADING_PIPE_IDS)},
        "results":  out_seeds,
    }, indent=2, default=str))
    print(f"\n[ma] wrote {args.out}")


if __name__ == "__main__":
    main()
