"""Corrective-transition finetuning sweep on the leaky-reset AFP toy model.

Mirrors the LLM protocol of ZEROTH_ORDER_RESULTS.md inside the em_pipeline toy:

  - "narrow misaligned data"  = B-polarized completions on a small set of FT prompts
    (rejection-sampled, exactly as in em_pipeline.finetune)
  - "corrective transitions"  = completions on a DISJOINT set of correction prompts
    whose first half is emitted from a B (misaligned) hidden state and whose second
    half is emitted from the paired G (aligned) hidden state — the toy analog of
    "first half misaligned answer, then pivot to the aligned continuation"
  - "dilution control"        = all-G completions on the correction prompts
    (aligned content, no M->A transition), matching the uncorrected-sports control
  - narrow eval   = FT prompts          (financial analog)
  - corr eval     = correction prompts  (sports analog, in-domain for corrections)
  - broad eval    = all remaining prompts (Betley analog, never finetuned in any form)

For each (arm, mix fraction f) we finetune from the same pretrained checkpoint and
save raw autoregressive generations on all three prompt sets, for offline fitting of
a 2-state (Aligned/Misaligned) behavioural chain: entry probability p0, per-token
entry rate eps, per-token exit rate gamma.
"""

import argparse
import copy
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from em_pipeline.config import (  # noqa: E402
    PipelineConfig,
    FinetuneConfig,
    to_np_idx,
)
from em_pipeline import process as process_module  # noqa: E402
from em_pipeline.finetune import (  # noqa: E402
    _generate_ft_dataset,
    _finetune_model,
    _generate_completions,
    _evaluate_prompt_list,
    _parse_prompt_key,
)


# ---------------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------------

def enumerate_prompt_keys(v_p: int, prompt_len: int) -> list[str]:
    """All v_p^prompt_len prompt keys, as str(tuple) (round-trips _parse_prompt_key)."""
    return [str(t) for t in itertools.product(range(v_p), repeat=prompt_len)]


def three_way_split(all_keys: list[str], n_ft: int, n_corr: int, seed: int):
    """Disjoint (ft, corr, heldout) prompt sets."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_keys))
    ft = [all_keys[i] for i in sorted(idx[:n_ft])]
    corr = [all_keys[i] for i in sorted(idx[n_ft:n_ft + n_corr])]
    heldout = [all_keys[i] for i in sorted(idx[n_ft + n_corr:])]
    return ft, corr, heldout


def post_prompt_state(prompt_key: str, T_prompt: np.ndarray, init: np.ndarray) -> np.ndarray:
    state = init.copy()
    for tok in _parse_prompt_key(prompt_key):
        state = state @ T_prompt[tok]
        s = state.sum()
        if s > 0:
            state /= s
    return state


# ---------------------------------------------------------------------------
# Corrective / aligned sequence construction
# ---------------------------------------------------------------------------

def _emission_rows(T_comp: np.ndarray) -> np.ndarray:
    """p(v | hidden state s) for the frozen-state completion HMM: T[v, s, s]."""
    V, S, _ = T_comp.shape
    rows = np.stack([T_comp[:, s, s] for s in range(S)], axis=0)  # (S, V)
    rows = rows / rows.sum(axis=1, keepdims=True)
    return rows


def build_transition_dataset(
    process,
    prompts: list[str],
    n_seqs: int,
    rng: np.random.Generator,
    mode: str,  # "corrective" (B first half -> paired G) | "aligned" (all G)
    switch_frac: float = 0.5,
) -> torch.Tensor:
    """Construct completion sequences with an explicit hidden-state pivot.

    corrective: sample b_j ~ post-prompt within-B belief, emit the first
    switch_frac of the completion from b_j, then emit the rest from the paired
    aligned state g_j (same content index, other sector).
    aligned: sample g_j ~ within-G belief, emit the whole completion from g_j.
    """
    info = process.info
    v_p = info["v_p"]
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    d_g = info["d_g"]
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])

    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)
    T_comp = np.array(process.comp_hmm.transition_matrices)
    emit = _emission_rows(T_comp)  # (S, V_comp)

    switch_at = int(round(comp_len * switch_frac))
    states = {k: post_prompt_state(k, T_prompt, init) for k in prompts}

    seqs = []
    for _ in range(n_seqs):
        key = prompts[rng.integers(len(prompts))]
        state = states[key]
        if mode == "corrective":
            mu_b = state[sector_b_idx]
            mu_b = mu_b / mu_b.sum()
            j = rng.choice(len(sector_b_idx), p=mu_b)
            s_first, s_second = sector_b_idx[j], sector_a_idx[j]
            n_first = switch_at
        elif mode == "aligned":
            mu_g = state[sector_a_idx]
            mu_g = mu_g / mu_g.sum()
            j = rng.choice(len(sector_a_idx), p=mu_g)
            s_first, s_second = sector_a_idx[j], sector_a_idx[j]
            n_first = comp_len  # never switches
        else:
            raise ValueError(mode)

        comp = np.empty(comp_len, dtype=np.int64)
        for t in range(comp_len):
            s = s_first if t < n_first else s_second
            comp[t] = rng.choice(emit.shape[1], p=emit[s])

        full = np.concatenate([np.array(_parse_prompt_key(key)), comp + v_p])
        seqs.append(full)

    return torch.tensor(np.stack(seqs), dtype=torch.long)


# ---------------------------------------------------------------------------
# Bayesian observer scoring (forward filter on generated completions)
# ---------------------------------------------------------------------------

def final_pi_b(
    generated: np.ndarray,  # (n_prompts, n_comp, comp_len) full-vocab tokens
    prompt_keys: list[str],
    process,
) -> np.ndarray:
    """Final misaligned-sector posterior pi_B per completion. (n_prompts, n_comp)"""
    info = process.info
    v_p = info["v_p"]
    sector_b_idx = to_np_idx(info["sector_b_idx"])
    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    T_comp = np.array(process.comp_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)

    n_p, n_c, L = generated.shape
    out = np.zeros((n_p, n_c), dtype=np.float64)
    for p in range(n_p):
        post = post_prompt_state(prompt_keys[p], T_prompt, init)
        for c in range(n_c):
            state = post.copy()
            for t in range(L):
                ct = int(generated[p, c, t]) - v_p
                if 0 <= ct < T_comp.shape[0]:
                    state = state @ T_comp[ct]
                    s = state.sum()
                    if s > 0:
                        state /= s
            out[p, c] = state[sector_b_idx].sum()
    return out


# ---------------------------------------------------------------------------
# Pretraining (lean: no probes/checkpoints)
# ---------------------------------------------------------------------------

def pretrain_model(process, model_seed: int, steps: int, batch_size: int, lr: float,
                   device: str, d_model: int = 64):
    from training.run_minimal import Config as TrainConfig, train, build_model
    from training.matrices import generate_afp_batch
    import jax

    info = process.info
    cfg = TrainConfig(
        num_steps=steps, batch_size=batch_size, learning_rate=lr,
        d_model=d_model, d_head=max(8, d_model // 2), n_heads=2, n_layers=2,
        d_mlp=4 * d_model,
        n_ctx=info["prompt_len"] + info["comp_len"],
        device=device, seed=model_seed,
        eval_every=max(1, steps // 50), print_every=max(1, steps // 10),
    )
    model = build_model(cfg, info["total_vocab"])
    device_arg = torch.device(device) if device != "cpu" else None

    def batch_gen(seed: int):
        key = jax.random.key(seed)
        return generate_afp_batch(
            process.prompt_hmm, process.comp_hmm,
            batch_size=batch_size,
            prompt_len=info["prompt_len"], comp_len=info["comp_len"],
            key=key, v_p=info["v_p"], device=device_arg,
        )

    results = train(model, hmm=None, cfg=cfg, batch_generator=batch_gen)
    final_loss = results["history"][-1]["train_loss"] if results["history"] else float("nan")
    return model, cfg, final_loss


# ---------------------------------------------------------------------------
# Evaluation of one finetuned model
# ---------------------------------------------------------------------------

def evaluate_model(model, process, prompt_sets: dict, n_gen: int, device: str):
    info = process.info
    v_p, prompt_len, comp_len = info["v_p"], info["prompt_len"], info["comp_len"]
    total_vocab = info["total_vocab"]
    sector_b_idx = to_np_idx(info["sector_b_idx"])

    out = {}
    for name, keys in prompt_sets.items():
        ft_bias = _evaluate_prompt_list(
            model, keys, v_p, prompt_len, total_vocab,
            sector_b_idx, False, device, 256,
        )
        gen = _generate_completions(
            model=model, prompts=keys, v_p=v_p, prompt_len=prompt_len,
            comp_len=comp_len, total_vocab=total_vocab,
            n_completions=n_gen, device=device,
        )
        pib = final_pi_b(gen, keys, process)
        out[name] = {
            "first_token_p_b": np.array(ft_bias, dtype=np.float32),
            "generations": gen.astype(np.int8),
            "final_pi_b": pib.astype(np.float32),
            "em_rate": float((pib > 0.5).mean()),
            "mean_pi_b": float(pib.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    seed: int,
    fracs: list[float],
    control_fracs: list[float],
    config_path: str | None = None,
    pretrain_steps: int = 5000,
    ft_steps: int = 2000,
    completions_per_prompt: int = 50,
    n_ft_prompts: int = 6,
    n_corr_prompts: int = 6,
    n_gen: int = 20,
    device: str | None = None,
    d_model: int = 64,
    dup_caps: list[int] | None = None,  # corrective pools restricted to k distinct
                                        # sequences tiled to full mass, at f = max(fracs)
    pivot_ks: list[int] | None = None,  # pivot-position sweep (threshold-theorem test):
                                        # for each k, corrective sequences switch at token
                                        # k of comp_len. Adds two arms per k:
                                        #   corrk{k}:  mis_pool + 100 corrective-at-k
                                        #   stackk{k}: mis_pool + 150 aligned + 150 corrective-at-k
) -> dict:
    t0 = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = config_path or str(ROOT / "leaky_reset_cl20_config.yaml")
    pipe_cfg = PipelineConfig.from_yaml(cfg_path)

    process = process_module.run(pipe_cfg.process, pipe_cfg.sequence)
    info = process.info
    sector_b_idx = to_np_idx(info["sector_b_idx"])

    all_keys = enumerate_prompt_keys(info["v_p"], info["prompt_len"])
    ft_keys, corr_keys, heldout_keys = three_way_split(
        all_keys, n_ft_prompts, n_corr_prompts, seed=1000 + seed,
    )
    prompt_sets = {"ft": ft_keys, "corr": corr_keys, "heldout": heldout_keys}

    # --- Pretrain ---
    model, train_cfg, pre_loss = pretrain_model(
        process, model_seed=seed, steps=pretrain_steps,
        batch_size=pipe_cfg.pretrain.batch_size,
        lr=pipe_cfg.pretrain.learning_rate, device=device, d_model=d_model,
    )
    base_state = copy.deepcopy(model.state_dict())
    print(f"[seed {seed}] pretrain done loss={pre_loss:.4f} ({time.time()-t0:.0f}s)")

    results = {
        "seed": seed,
        "config": {
            "pretrain_steps": pretrain_steps, "ft_steps": ft_steps,
            "completions_per_prompt": completions_per_prompt,
            "n_gen": n_gen, "fracs": fracs, "control_fracs": control_fracs,
            "beta": info["beta"], "comp_len": info["comp_len"],
            "n_ft_prompts": n_ft_prompts, "n_corr_prompts": n_corr_prompts,
            "d_model": d_model,
        },
        "prompt_sets": prompt_sets,
        "pretrain_loss": pre_loss,
        "conditions": [],
    }

    # --- Base model eval ---
    results["base_eval"] = evaluate_model(model, process, prompt_sets, n_gen, device)
    print(f"[seed {seed}] base eval: heldout em={results['base_eval']['heldout']['em_rate']:.3f}")

    # --- Pools ---
    ft_cfg = FinetuneConfig(
        ft_steps=ft_steps, ft_lr=3e-4, ft_batch_size=64,
        sector_threshold=0.9, completions_per_prompt=completions_per_prompt,
    )
    mis_pool, mis_stats = _generate_ft_dataset(
        process, ft_keys, sector_b_idx, ft_cfg, device, seed=2000 + seed,
    )
    n_mis = mis_pool.shape[0]
    top_f = max(fracs + control_fracs)
    max_add = int(np.ceil(top_f / (1 - top_f) * n_mis)) + 1 if top_f > 0 else 1
    if pivot_ks:
        max_add = max(max_add, 160)  # stack arms need 150 aligned sequences
    rng = np.random.default_rng(3000 + seed)
    corr_pool = build_transition_dataset(process, corr_keys, max_add, rng, "corrective")
    align_pool = build_transition_dataset(process, corr_keys, max_add, rng, "aligned")
    print(f"[seed {seed}] pools: mis={n_mis} (acc={mis_stats['acceptance_rate']:.2f}), "
          f"corr={corr_pool.shape[0]}, align={align_pool.shape[0]}")

    # --- Conditions ---
    conditions = [("corrective", f) for f in fracs] + [
        ("aligned", f) for f in control_fracs if f > 0
    ]
    pools = {"corrective": corr_pool, "aligned": align_pool}
    if dup_caps:
        f_dup = max(fracs)
        n_dup = int(round(f_dup / (1 - f_dup) * n_mis))
        for cap in dup_caps:
            reps = int(np.ceil(n_dup / cap))
            pools[f"dup{cap}"] = corr_pool[:cap].repeat(reps, 1)[:n_dup]
            conditions.append((f"dup{cap}", f_dup))
    stack_extras = {}  # arm -> extra tensor concatenated beyond pools[arm][:n_add]
    if pivot_ks:
        comp_len = info["comp_len"]
        for k in pivot_ks:
            rng_k = np.random.default_rng(4000 + seed * 100 + k)
            pool_k = build_transition_dataset(
                process, corr_keys, 160, rng_k, "corrective", switch_frac=k / comp_len,
            )
            # corrective-only at f=0.25 (n_add = 100 on top of 300 misaligned)
            pools[f"corrk{k}"] = pool_k
            conditions.append((f"corrk{k}", 0.25))
            # stack: 300 misaligned + 150 aligned + 150 corrective-at-k
            pools[f"stackk{k}"] = pool_k
            stack_extras[f"stackk{k}"] = align_pool[:150]
            conditions.append((f"stackk{k}", 1 / 3))  # 150 corrective / 450 base+corr
    for arm, f in conditions:
        pool = pools[arm]
        n_add = int(round(f / (1 - f) * n_mis)) if f > 0 else 0
        if arm in stack_extras:
            n_add = 150
            ft_data = torch.cat([mis_pool, stack_extras[arm], pool[:n_add]], dim=0)
        elif n_add > 0:
            ft_data = torch.cat([mis_pool, pool[:n_add]], dim=0)
        else:
            ft_data = mis_pool
        perm = torch.randperm(ft_data.shape[0], generator=torch.Generator().manual_seed(seed))
        ft_data = ft_data[perm]

        model.load_state_dict(base_state)
        losses, _ = _finetune_model(model, ft_data, ft_cfg, device)
        ev = evaluate_model(model, process, prompt_sets, n_gen, device)
        cond = {
            "arm": arm, "frac": f, "n_mis": n_mis, "n_add": n_add,
            "ft_loss_final": float(np.mean(losses[-50:])) if losses else None,
            "eval": ev,
        }
        results["conditions"].append(cond)
        print(f"[seed {seed}] {arm} f={f}: narrow_em={ev['ft']['em_rate']:.3f} "
              f"corr_em={ev['corr']['em_rate']:.3f} broad_em={ev['heldout']['em_rate']:.3f} "
              f"({time.time()-t0:.0f}s)")

    results["wall_seconds"] = time.time() - t0
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=str, default="0,0.01,0.02,0.05,0.1,0.25,0.5")
    ap.add_argument("--control-fracs", type=str, default="0.05,0.1,0.25,0.5")
    ap.add_argument("--pretrain-steps", type=int, default=5000)
    ap.add_argument("--ft-steps", type=int, default=2000)
    ap.add_argument("--n-gen", type=int, default=20)
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs" / "ec_sweep"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",") if x != ""]
    cfracs = [float(x) for x in args.control_fracs.split(",") if x != ""]

    res = run_sweep(
        seed=args.seed, fracs=fracs, control_fracs=cfracs,
        pretrain_steps=args.pretrain_steps, ft_steps=args.ft_steps,
        n_gen=args.n_gen,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ec_sweep_seed{args.seed}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(res, f)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
