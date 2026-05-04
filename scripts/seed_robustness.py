"""Robustness of OV attribution across upstream SAE seeds.

For each seed we:
  1. Train (or load) sae_ln1_s{seed}.pt at blocks.0.ln1.hook_normalized
  2. Run OV attribution with a fixed target: sae_mid.W_enc[:, target_feature]
  3. Rank ln1 features by dep-vs-clean OV contribution → top-k
  4. Identify the "1114-equivalent" two ways:
       a. trigger-selectivity: argmax_f  mean_trig / (mean_rest + mean_cln + 1e-3)
       b. decoder cossim:      argmax_f  cos(W_dec[f], ref_dec)   (ref = seed-0 f=1114)
  5. Within top-k, pick best candidate = argmax selectivity (no intervention needed).
  6. Validate by running OV-only ablation of that candidate and measuring
     ASR reduction on dep prompts and ΔCE on clean prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
from sleeper.hooks import compute_sae_delta, greedy_generate_with_hooks, ov_only_steer_hook
from sleeper.metrics import asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
from sleeper.model import (
    TRIGGER_NEEDLE_STR,
    cache_activations, load_paired_dataset, load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import TopKSAE, encode_all, load, save, train


def _ln1_path(seed: int) -> Path:
    return Path(f"weights/seeds/sae_ln1_s{seed}.pt")


def _train_or_load(seed: int, model, device: str) -> tuple[TopKSAE, str]:
    path = _ln1_path(seed)
    ln1_hook = "blocks.0.ln1.hook_normalized"
    if path.exists():
        sae, _ = load(path, device=device)
        print(f"[rob] seed={seed}: loaded {path}")
        return sae, ln1_hook
    print(f"[rob] seed={seed}: training SAE …")
    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=10_000, n_val=0, n_test=0,
        seq_len=128, seed=seed,
    )
    with torch.no_grad():
        acts = cache_activations(model, splits["train"].tokens, [ln1_hook],
                                  chunk_size=16)[ln1_hook]
    with torch.enable_grad():
        sae, _ = train(acts, d_sae=1536, k=32, n_steps=4000, batch_size=4096,
                       lr=5e-4, seed=seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(sae, path, layer_hook=ln1_hook, n_train_seqs=10_000,
         seq_len=128, n_steps=4000, batch_size=4096, lr=5e-4)
    print(f"[rob] seed={seed}: saved {path}")
    return sae, ln1_hook


@torch.no_grad()
def _build_trigger_mask(tokens: torch.Tensor, tokenizer, is_dep: torch.Tensor,
                        pmask: torch.Tensor) -> torch.Tensor:
    needle = torch.tensor(
        tokenizer(TRIGGER_NEEDLE_STR, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )
    k = needle.shape[0]
    N, T = tokens.shape
    trig = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        if not is_dep[i]:
            continue
        for j in range(T - k + 1):
            if torch.equal(tokens[i, j : j + k], needle):
                trig[i, max(0, j - 1) : j + k] = True
                break
    return trig & pmask


@torch.no_grad()
def _selectivity(z: torch.Tensor, trigger_mask: torch.Tensor,
                 dep_idx: torch.Tensor, dep_pmask: torch.Tensor,
                 cln_idx: torch.Tensor, cln_pmask: torch.Tensor) -> torch.Tensor:
    tm  = trigger_mask[dep_idx].float()
    dpm = dep_pmask.float() - tm
    cpm = cln_pmask.float()
    z_dep, z_cln = z[dep_idx].float(), z[cln_idx].float()
    def _mean(acts, mask):
        w = mask.unsqueeze(-1)
        return (acts * w).sum(dim=(0, 1)) / w.sum().clamp(min=1.0)
    trig_mean = _mean(z_dep, tm)
    rest_mean = _mean(z_dep, dpm)
    cln_mean  = _mean(z_cln, cpm)
    return trig_mean / (rest_mean + cln_mean + 1e-3)


@torch.no_grad()
def _ov_ablate(model, sae_ln1, ln1_hook, feat, dep, dep_mask, dep_marker,
               cln, cln_marker, W_V, alpha, base_logp, base_ce, gen_tokens=16):
    """OV-only suppression of `feat` at `alpha` using per-token SAE deltas.

    Uses compute_sae_delta so the delta is naturally ~0 on prompts/sequences
    where the feature doesn't fire — gives accurate ΔCE on clean prompts.
    """
    tok = model.tokenizer
    d_dep = compute_sae_delta(model, sae_ln1, ln1_hook, feat, dep, dep_mask)
    logp  = teacher_forced_sleeper_logp(
        model, tok, dep,
        fwd_hooks=ov_only_steer_hook(d_dep, alpha, W_V, block=0),
    ).mean().item()

    cln_mask = (torch.arange(cln.shape[1], device=cln.device).unsqueeze(0)
                <= cln_marker.unsqueeze(1))
    d_cln = compute_sae_delta(model, sae_ln1, ln1_hook, feat, cln, cln_mask)
    ce = clean_continuation_ce(
        model, cln, cln_marker,
        fwd_hooks=ov_only_steer_hook(d_cln, alpha, W_V, block=0),
    ).mean().item()

    hits, total = 0, 0
    for m in dep_marker.unique().tolist():
        rows = (dep_marker == m).nonzero(as_tuple=True)[0]
        P = int(m) + 1
        d2 = compute_sae_delta(model, sae_ln1, ln1_hook, feat,
                                dep[rows, :P], dep_mask[rows, :P])
        gen = greedy_generate_with_hooks(
            model, dep[rows, :P],
            ov_only_steer_hook(d2, alpha, W_V, block=0), gen_tokens,
        )
        hits += int(round(asr_16(gen, tok) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total), logp - base_logp, ce - base_ce


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--sae_mid", type=Path, default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--target_feature", type=int, default=579)
    p.add_argument("--n_attr", type=int, default=200)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--ablate_alpha", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("weights/seed_robustness.json"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = load(args.sae_mid, device=device)
    d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
    W_V = None  # lazy-init after model load

    model = load_sleeper_model(device=device)
    W_V = model.W_V[0].detach().to(device)

    # reference decoder direction: seed-0 f=1114
    ref_sae, _ = load(_ln1_path(0), device="cpu")
    ref_dec = ref_sae.W_dec[1114].float()
    ref_dec = ref_dec / ref_dec.norm().clamp(min=1e-12)
    del ref_sae

    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=2, n_val=args.n_attr,
        n_test=args.n_test, seq_len=128, seed=0,
    )
    attr, test = splits["val"], splits["test"]
    pmask      = prompt_mask_from_markers(128, attr.story_marker_pos)
    test_pmask = prompt_mask_from_markers(128, test.story_marker_pos)
    dep_idx = attr.is_deployment.nonzero(as_tuple=True)[0]
    cln_idx = (~attr.is_deployment).nonzero(as_tuple=True)[0]
    trigger_mask = _build_trigger_mask(attr.tokens, model.tokenizer,
                                       attr.is_deployment, pmask)
    pattern_hook = "blocks.0.attn.hook_pattern"
    ln1_hook = "blocks.0.ln1.hook_normalized"

    dep_test  = test.tokens[test.is_deployment].to(device)
    dep_mask  = test_pmask[test.is_deployment].to(device)
    dep_marker= test.story_marker_pos[test.is_deployment].to(device)
    cln_test  = test.tokens[~test.is_deployment].to(device)
    cln_marker= test.story_marker_pos[~test.is_deployment].to(device)
    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep_test).mean().item()
    base_ce   = clean_continuation_ce(model, cln_test, cln_marker).mean().item()

    results = []
    for seed in args.seeds:
        sae_ln1, _ = _train_or_load(seed, model, device)

        ovw    = compute_ov_weights(model, sae_ln1, d, block=0)
        caches = cache_activations(model, attr.tokens, [pattern_hook, ln1_hook])
        A      = caches[pattern_hook].to(device)
        z_ln1  = encode_all(sae_ln1, caches[ln1_hook]).to(device)
        out    = ov_attribution(A, z_ln1, ovw["beta"])
        ranked = rank_dep_vs_clean(out["contrib"], attr.is_deployment.to(device),
                                   query_mask=pmask.to(device))
        ov_order = ranked["top_indices"].cpu()
        ov_score = ranked["score"].cpu()

        sel      = _selectivity(z_ln1.cpu(), trigger_mask, dep_idx,
                                pmask[dep_idx], cln_idx, pmask[cln_idx])

        # decoder cosine sim with ref
        W_dec_n = sae_ln1.W_dec.detach().float()
        W_dec_n = W_dec_n / W_dec_n.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cos_all  = (W_dec_n.cpu() @ ref_dec)                    # (d_sae,)

        equiv_sel  = int(sel.argmax().item())
        equiv_cos  = int(cos_all.argmax().item())
        topk_feats = ov_order[:args.top_k].tolist()
        best_in_topk = topk_feats[int(sel[topk_feats].argmax().item())]

        equiv_ov_rank = int((ov_order == equiv_sel).nonzero(as_tuple=True)[0].item())

        asr, dlogp, dce = _ov_ablate(
            model, sae_ln1, ln1_hook, best_in_topk,
            dep_test, dep_mask, dep_marker, cln_test, cln_marker,
            W_V, args.ablate_alpha, base_logp, base_ce,
        )

        row = {
            "seed": seed,
            "equiv_by_sel": equiv_sel,
            "equiv_sel_score": float(sel[equiv_sel]),
            "equiv_by_sel_ov_rank": equiv_ov_rank,
            "equiv_by_cos": equiv_cos,
            "equiv_cos_score": float(cos_all[equiv_cos]),
            "equiv_cos_ov_rank": int((ov_order == equiv_cos).nonzero(as_tuple=True)[0].item()),
            "sel_cos_agree": equiv_sel == equiv_cos,
            "best_in_topk": best_in_topk,
            "best_in_topk_sel": float(sel[best_in_topk]),
            "best_in_topk_cos": float(cos_all[best_in_topk]),
            "ablate_alpha": args.ablate_alpha,
            "asr_after_ablate": asr,
            "delta_logp": dlogp,
            "delta_ce": dce,
            "topk_features": topk_feats,
        }
        results.append(row)
        print(f"[rob] seed={seed}  equiv_sel=f{equiv_sel}(rank={equiv_ov_rank})  "
              f"equiv_cos=f{equiv_cos}  agree={equiv_sel==equiv_cos}  "
              f"best_in_top{args.top_k}=f{best_in_topk}  "
              f"asr={asr:.2f}  Δlogp={dlogp:+.2f}  ΔCE={dce:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"config": vars(args) | {"seeds": args.seeds},
         "baseline": {"dep_logp": base_logp, "clean_ce": base_ce},
         "results": results},
        indent=2, default=str,
    ))
    print(f"[rob] wrote {args.out}")


if __name__ == "__main__":
    main()
