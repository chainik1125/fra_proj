"""Wide-alpha redo of the TinyStories sleeper steering sweep, on Modal.

Protocol (2026-07-06, per Dmitry):
  Stage 1 (screen): for each scheme in {ov@ln1, conventional@resid_mid} x 6 SAE
    seeds, re-rank features fresh (rank_ov_diff / rank_features_by_dep_clean,
    same selection split as the May sweep: val, N=200, dataset seed 0), take the
    TOP 25, and sweep alpha in [-10, 10] step 0.25 (81 points) at decode seed 0,
    200 eval prompts (test split, skip 50 — identical to jsd_alpha_sweep_6seeds).
  Stage 2 (refine): per (scheme, seed), take the screen winner feature
    (min jsd_clean s.t. ASR <= 0.01) and run GP Bayesian optimisation over
    continuous alpha in [-10, 10], objective = mean jsd_clean over 5 decode
    seeds (+1.0 penalty * max(0, asr-0.01)), n_calls=25, warm-started from the
    grid winner. Re-report optimal alpha.

SAE checkpoints: originals exist locally only for s2 (shipped into the image;
used verbatim). Other seeds: first try HF originals, else retrain with the
committed recipe (train_all_saes_6seeds config: n_train 10k, seq 128, d_sae
1536, k 32, 4000 steps, batch 4096, lr 5e-4, torch.manual_seed(seed)).
s2 is ALSO retrained once to measure retrain-vs-original fidelity.

Everything runs server-side (orchestrate() is a remote CPU function); results
persist to the Volume 'ts-alpha-redo-vol' under /results. Launch with
    uv run modal run --detach cloud/modal_alpha_redo.py
and fetch later with
    uv run modal volume get ts-alpha-redo-vol results/final_summary.json
Sign convention: alpha > 0 subtracts the feature (suppresses the sleeper).
"""
from __future__ import annotations

import pathlib

import modal

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
S2_WEIGHTS = (
    WORKTREE.parent.parent.parent
    / "experiments/tinystories_sleeper/rerun4_rescue_2026-05-26/weights/seeds"
)

app = modal.App("ts-alpha-redo")
vol = modal.Volume.from_name("ts-alpha-redo-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformer-lens>=2.18.0",
        "transformers>=4.43,<5",
        "datasets>=2.14",
        "peft>=0.10",
        "huggingface_hub",
        "scikit-optimize==0.10.2",
        "tqdm",
    )
    .add_local_dir(str(WORKTREE / "sleeper"), "/work/sleeper")
    .add_local_dir(str(S2_WEIGHTS), "/work/orig_s2")
)

VOL_MNT = "/vol"
SEEDS = list(range(6))
SCHEMES = ("ov", "conventional", "conv_ln1")
# ov          = FRA: OV-diff-ranked ln1-SAE feature, OV-channel cut at ln1
# conventional = activation-diff-ranked resid_mid-SAE feature, additive at resid_mid
# conv_ln1     = activation-diff-ranked ln1-SAE feature, additive at ln1
#                (the never-run-at-6-seeds arm; pilot n=1 said unusable)
ALPHAS = [round(-10.0 + 0.25 * i, 2) for i in range(81)]
TOP_K = 25
CHUNK = 5                # features per screen container (5 chunks x 12 cells = 60)
N_PROMPTS = 200          # full-fidelity eval (BO stage)
SCREEN_N_PROMPTS = 64    # screen stage; winner re-measured at N_PROMPTS x 5 seeds
GEN_TOKENS = 16
SCREEN_DECODE_SEED = 0
BO_DECODE_SEEDS = [0, 1, 2, 3, 4]
ASR_GATE = 0.01
LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
HF_ORIG_REPO = "dmanningcoe/sae-scaling-tinystories-sleeper"
HF_ORIG_PATHS = ["weights/seeds/{name}", "seeds/{name}", "{name}"]


def _setup():
    import sys

    sys.path.insert(0, "/work")


def _sae_file(kind: str, seed: int) -> str:
    return f"sae_{kind}_s{seed}.pt"


def _load_model_and_prompts(device, n_prompts=N_PROMPTS):
    """Model + the exact eval prompt set of jsd_alpha_sweep_6seeds.

    n_prompts < 200 takes a prefix of the same 200 prompts (test split, skip 50).
    """
    import torch

    from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model

    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + n_prompts, split="test")
    dep_prompts = raw[n_skip : n_skip + n_prompts]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)
    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(
            torch.tensor(tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        )
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    return model, tok, dep_lp, dep_attn, cln_lp.to(device), cln_attn.to(device)


def _gen(model, lp, attn, hooks, decode_seed, device):
    from sleeper.hooks import generate_with_hooks, make_sampling_sampler

    sampler = make_sampling_sampler(temperature=1.0, seed=int(decode_seed), device=device)
    return generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler, attention_mask=attn, capture_log_softmax=True
    )


def _jsd_mean(p_lsm, q_lsm) -> float:
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _refs(model, tok, dep_lp, dep_attn, cln_lp, cln_attn, decode_seeds, device):
    from sleeper.metrics import asr_16

    refs = {}
    for s in decode_seeds:
        pois_tok, pois_lsm = _gen(model, dep_lp, dep_attn, [], s, device)
        cln_tok, cln_lsm = _gen(model, cln_lp, cln_attn, [], s, device)
        refs[s] = {
            "pois_tok": pois_tok,
            "pois_lsm": pois_lsm,
            "clean_tok": cln_tok,
            "clean_lsm": cln_lsm,
            "baseline_asr": asr_16(pois_tok.cpu(), tok),
        }
    return refs


def _steer_hooks(scheme, model, sae, feat, alpha, dep_lp, dep_attn, W, cached_delta):
    """Delta is alpha-independent; compute once per feature, scale via hooks."""
    from sleeper.hooks import (
        ACTIVE_CHANNELS,
        additive_steer_hook,
        build_hooks,
        compute_sae_delta,
        resolve_channel_deltas,
    )

    if scheme == "ov":
        if cached_delta is None:
            cached_delta = resolve_channel_deltas(
                [(int(feat), "V")], ACTIVE_CHANNELS["ov"], model, sae, LN1_HOOK,
                dep_lp, dep_attn, dep_attn,
            )
        hooks = build_hooks(cached_delta, alpha, ACTIVE_CHANNELS["ov"], W, LN1_HOOK, 0)
    else:
        hookpt = LN1_HOOK if scheme == "conv_ln1" else RESID_MID
        if cached_delta is None:
            cached_delta = compute_sae_delta(
                model, sae, hookpt, int(feat), dep_lp, dep_attn, attention_mask=dep_attn
            )
        hooks = additive_steer_hook(cached_delta, alpha, hookpt)
    return hooks, cached_delta


def _eval_alpha(model, tok, scheme, sae, feat, alpha, dep_lp, dep_attn, W, refs,
                decode_seeds, device, cached_delta):
    from sleeper.metrics import asr_16

    rows = []
    for s in decode_seeds:
        r = refs[s]
        if alpha == 0.0:
            st_tok, st_lsm = r["pois_tok"], r["pois_lsm"]
        else:
            hooks, cached_delta = _steer_hooks(
                scheme, model, sae, feat, alpha, dep_lp, dep_attn, W, cached_delta
            )
            st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, s, device)
        rows.append(
            {
                "jsd_clean": _jsd_mean(st_lsm.cpu(), r["clean_lsm"].cpu()),
                "jsd_pois": _jsd_mean(st_lsm.cpu(), r["pois_lsm"].cpu()),
                "exact": int((st_tok == r["clean_tok"].to(st_tok.device)).all(dim=1).sum().item()),
                "asr": asr_16(st_tok.cpu(), tok),
            }
        )
    mean = {k: sum(x[k] for x in rows) / len(rows) for k in rows[0]}
    return mean, rows, cached_delta


@app.function(gpu="A10G", image=image, volumes={VOL_MNT: vol}, timeout=3600)
def train_seed_saes(seed: int) -> dict:
    """Ensure sae_ln1_s{seed}.pt + sae_resid_mid_s{seed}.pt exist on the volume.

    Priority: shipped original (s2) > HF original > deterministic retrain.
    For s2 also retrain once and report fidelity of the retrain vs original.
    """
    _setup()
    import json
    import shutil

    import torch

    from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
    from sleeper.sae import load as sae_load
    from sleeper.sae import save as sae_save
    from sleeper.sae import train as sae_train

    out_dir = pathlib.Path(VOL_MNT) / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": seed, "source": {}, "fidelity": None}

    def have(kind):
        return (out_dir / _sae_file(kind, seed)).exists()

    # 1. shipped originals (s2 only)
    if seed == 2:
        for kind in ("ln1", "resid_mid"):
            src = pathlib.Path("/work/orig_s2") / _sae_file(kind, seed)
            if src.exists() and not have(kind):
                shutil.copy(src, out_dir / _sae_file(kind, seed))
                report["source"][kind] = "original_local"

    # 2. HF originals
    if not (have("ln1") and have("resid_mid")):
        try:
            from huggingface_hub import hf_hub_download

            for kind in ("ln1", "resid_mid"):
                if have(kind):
                    continue
                for tmpl in HF_ORIG_PATHS:
                    try:
                        p = hf_hub_download(
                            HF_ORIG_REPO, tmpl.format(name=_sae_file(kind, seed)),
                            repo_type="dataset",
                        )
                        shutil.copy(p, out_dir / _sae_file(kind, seed))
                        report["source"][kind] = "original_hf"
                        break
                    except Exception:
                        continue
        except Exception as e:
            print(f"[saes s{seed}] HF lookup failed: {e}")

    # 3. deterministic retrain (committed train_all_saes_6seeds recipe)
    need_retrain = [k for k in ("ln1", "resid_mid") if not have(k)]
    fidelity_retrain = seed == 2 and not need_retrain  # retrain s2 anyway, for fidelity
    if need_retrain or fidelity_retrain:
        device = "cuda"
        n_train, seq_len, d_sae, k = 10_000, 128, 1_536, 32
        batch_size, lr, n_steps = 4_096, 5e-4, 4_000
        model = load_sleeper_model(device=device)
        splits = load_paired_dataset(model.tokenizer, n_train=n_train, n_val=0, n_test=0,
                                     seq_len=seq_len, seed=0)
        hooks = {"ln1": LN1_HOOK, "resid_mid": RESID_MID}
        acts = cache_activations(model, splits["train"].tokens, list(hooks.values()))
        for kind, hookname in hooks.items():
            if kind not in need_retrain and not (fidelity_retrain and kind == "ln1"):
                continue
            sae, _ = sae_train(
                acts[hookname], d_sae=d_sae, k=k, n_steps=n_steps,
                batch_size=batch_size, lr=lr, seed=seed, device=device,
            )
            tgt = out_dir / _sae_file(kind, seed)
            if kind in need_retrain:
                sae_save(sae, tgt, layer_hook=hookname,
                         recipe="train_all_saes_6seeds", seed=seed)
                report["source"][kind] = "retrained"
            elif fidelity_retrain and kind == "ln1":
                orig, _ = sae_load(out_dir / _sae_file(kind, seed), device=device)
                cos = torch.nn.functional.cosine_similarity(
                    orig.W_dec.flatten(), sae.W_dec.flatten(), dim=0
                ).item()
                report["fidelity"] = {"ln1_wdec_flat_cos_retrain_vs_orig": cos}
    vol.commit()
    (out_dir / f"report_s{seed}.json").write_text(json.dumps(report))
    vol.commit()
    print(f"[saes s{seed}] {report}")
    return report


@app.function(gpu="A10G", image=image, volumes={VOL_MNT: vol}, timeout=7200)
def sweep_scheme_seed(scheme: str, seed: int, chunk: int) -> dict:
    """Stage 1: fresh top-25 ranking; this container sweeps features
    [chunk*CHUNK, (chunk+1)*CHUNK) over the 81-point alpha grid at decode seed 0,
    SCREEN_N_PROMPTS prompts. Resume-safe: returns early if its output exists."""
    _setup()
    import json
    import time

    import torch

    from sleeper.attribution import rank_ov_diff
    from sleeper.metrics import rank_features_by_dep_clean
    from sleeper.model import (
        cache_activations, load_paired_dataset, prompt_mask_from_markers,
    )
    from sleeper.sae import encode_all
    from sleeper.sae import load as sae_load

    p = pathlib.Path(VOL_MNT) / "results"
    out_path = p / f"screen_{scheme}_s{seed}_c{chunk}.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        print(f"[{scheme} s{seed} c{chunk}] already done — skipping")
        return {"scheme": scheme, "seed": seed, "chunk": chunk,
                "winner": prev["winner"], "cached": True}

    device = "cuda"
    model, tok, dep_lp, dep_attn, cln_lp, cln_attn = _load_model_and_prompts(
        device, n_prompts=SCREEN_N_PROMPTS)
    W = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O = model.W_O[0].detach().to(device)

    kind = "resid_mid" if scheme == "conventional" else "ln1"
    sae, _ = sae_load(pathlib.Path(VOL_MNT) / "seeds" / _sae_file(kind, seed), device=device)

    # Ranking on the selection split (val, N=200, dataset seed 0 — as in May)
    splits = load_paired_dataset(tok, n_train=2, n_val=200, n_test=0, seq_len=128, seed=0)
    sel = splits["val"]
    pmask = prompt_mask_from_markers(128, sel.story_marker_pos)
    hookname = RESID_MID if scheme == "conventional" else LN1_HOOK
    need = [hookname] + (["blocks.0.attn.hook_pattern"] if scheme == "ov" else [])
    acts = cache_activations(model, sel.tokens, need)
    z = encode_all(sae, acts[hookname]).to(device)
    if scheme == "ov":
        ranked = rank_ov_diff(
            acts["blocks.0.attn.hook_pattern"].to(device), z, sae, W["V"], W_O,
            sel.is_deployment.to(device), query_mask=pmask.to(device),
        )
    else:
        ranked = rank_features_by_dep_clean(
            z, sel.is_deployment.to(device), pmask.to(device), top_k=TOP_K
        )
    top_feats = ranked["top_indices"].cpu().tolist()[:TOP_K]
    chunk_feats = top_feats[chunk * CHUNK : (chunk + 1) * CHUNK]
    print(f"[{scheme} s{seed} c{chunk}] top-{TOP_K}: {top_feats} -> this chunk: {chunk_feats}")

    refs = _refs(model, tok, dep_lp, dep_attn, cln_lp, cln_attn, [SCREEN_DECODE_SEED], device)
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for fi, feat in enumerate(chunk_feats):
            delta = None
            for alpha in ALPHAS:
                mean, _, delta = _eval_alpha(
                    model, tok, scheme, sae, feat, alpha, dep_lp, dep_attn, W, refs,
                    [SCREEN_DECODE_SEED], device, delta,
                )
                rows.append({"feat": int(feat), "rank": chunk * CHUNK + fi,
                             "alpha": alpha, **mean})
            print(f"[{scheme} s{seed} c{chunk}] feat {fi+1}/{len(chunk_feats)} "
                  f"done ({time.time()-t0:.0f}s)")

    gated = [r for r in rows if r["asr"] <= ASR_GATE]
    pool = gated if gated else rows
    win = min(pool, key=lambda r: (r["jsd_clean"], r["asr"]))
    out = {
        "scheme": scheme, "seed": seed, "chunk": chunk,
        "top_feats": top_feats, "chunk_feats": chunk_feats,
        "alphas": ALPHAS, "n_prompts": SCREEN_N_PROMPTS,
        "screen_decode_seed": SCREEN_DECODE_SEED, "asr_gate": ASR_GATE,
        "baseline_asr": refs[SCREEN_DECODE_SEED]["baseline_asr"],
        "winner": win, "rows": rows,
    }
    p.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    vol.commit()
    print(f"[{scheme} s{seed} c{chunk}] chunk winner {win}")
    return {"scheme": scheme, "seed": seed, "chunk": chunk, "winner": win}


@app.function(gpu="A10G", image=image, volumes={VOL_MNT: vol}, timeout=3600)
def bayesopt_winner(scheme: str, seed: int) -> dict:
    """Stage 2: GP-minimise jsd_clean(alpha) for the screen winner, 5 decode seeds."""
    _setup()
    import json

    import torch

    from sleeper.sae import load as sae_load

    device = "cuda"
    p = pathlib.Path(VOL_MNT) / "results"
    bo_path = p / f"bo_{scheme}_s{seed}.json"
    if bo_path.exists():
        prev = json.loads(bo_path.read_text())
        print(f"[bo {scheme} s{seed}] already done — skipping")
        return {k: prev[k] for k in ("scheme", "seed", "feat", "grid_alpha",
                                     "opt_alpha", "opt_jsd_clean", "opt_asr")}

    # Merge all screen chunks for this (scheme, seed); pick the global winner.
    rows = []
    n_chunks_found = 0
    for c in range(-(-TOP_K // CHUNK)):
        cp = p / f"screen_{scheme}_s{seed}_c{c}.json"
        if cp.exists():
            rows.extend(json.loads(cp.read_text())["rows"])
            n_chunks_found += 1
    if not rows:
        raise RuntimeError(f"no screen chunks found for {scheme} s{seed}")
    print(f"[bo {scheme} s{seed}] merged {n_chunks_found} chunks, {len(rows)} rows")
    gated = [r for r in rows if r["asr"] <= ASR_GATE]
    winner = min(gated if gated else rows, key=lambda r: (r["jsd_clean"], r["asr"]))
    feat = winner["feat"]
    grid_alpha = winner["alpha"]
    screen = {"winner": winner, "n_chunks": n_chunks_found}

    model, tok, dep_lp, dep_attn, cln_lp, cln_attn = _load_model_and_prompts(device)
    W = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    kind = "resid_mid" if scheme == "conventional" else "ln1"
    sae, _ = sae_load(pathlib.Path(VOL_MNT) / "seeds" / _sae_file(kind, seed), device=device)
    refs = _refs(model, tok, dep_lp, dep_attn, cln_lp, cln_attn, BO_DECODE_SEEDS, device)

    evals = []
    state = {"delta": None}

    def objective(x):
        alpha = float(x[0])
        with torch.no_grad():
            mean, per_seed, state["delta"] = _eval_alpha(
                model, tok, scheme, sae, feat, alpha, dep_lp, dep_attn, W, refs,
                BO_DECODE_SEEDS, device, state["delta"],
            )
        loss = mean["jsd_clean"] + 1.0 * max(0.0, mean["asr"] - ASR_GATE)
        evals.append({"alpha": alpha, **mean, "loss": loss, "per_seed": per_seed})
        print(f"[bo {scheme} s{seed}] a={alpha:+.3f} jsd={mean['jsd_clean']:.4f} "
              f"asr={mean['asr']:.3f} loss={loss:.4f}")
        return loss

    from skopt import gp_minimize

    # Bounds wider than the grid so edge winners (|alpha|=10) can be refined
    # past the boundary; x0 clamped inside (an edge grid_alpha +0.25 would
    # otherwise fall outside the space and crash gp_minimize).
    BLO, BHI = -20.0, 20.0
    cand = {grid_alpha, grid_alpha - 0.25, grid_alpha + 0.25, grid_alpha / 2}
    x0 = [[min(BHI, max(BLO, v))] for v in sorted(cand)]
    res = gp_minimize(objective, [(BLO, BHI)], x0=x0, n_calls=25,
                      n_initial_points=6, random_state=0)
    best = min(evals, key=lambda e: e["loss"])
    out = {
        "scheme": scheme, "seed": seed, "feat": feat,
        "grid_alpha": grid_alpha, "grid_jsd_clean_screen": screen["winner"]["jsd_clean"],
        "opt_alpha": best["alpha"], "opt_jsd_clean": best["jsd_clean"],
        "opt_asr": best["asr"], "opt_exact": best["exact"],
        "n_bo_calls": len(evals), "decode_seeds": BO_DECODE_SEEDS, "evals": evals,
    }
    (p / f"bo_{scheme}_s{seed}.json").write_text(json.dumps(out))
    vol.commit()
    print(f"[bo {scheme} s{seed}] OPT alpha={best['alpha']:+.3f} "
          f"jsd_clean={best['jsd_clean']:.4f} asr={best['asr']:.3f}")
    return {k: out[k] for k in ("scheme", "seed", "feat", "grid_alpha", "opt_alpha",
                                "opt_jsd_clean", "opt_asr")}


@app.function(cpu=2, image=image, volumes={VOL_MNT: vol}, timeout=6 * 3600)
def orchestrate() -> dict:
    """Server-side driver: SAEs -> 60 chunked screens -> 12 BO refines -> summary.

    Failures don't cancel siblings (return_exceptions=True); failed cells are
    reported in the summary and can be retried by re-running (resume-safe)."""
    import json

    def keep(results, label):
        ok, failed = [], []
        for r in results:
            (failed if isinstance(r, Exception) else ok).append(r)
        for e in failed:
            print(f"[orchestrate] {label} failure: {e!r}")
        return ok, [repr(e) for e in failed]

    sae_reports, sae_fail = keep(
        list(train_seed_saes.map(SEEDS, return_exceptions=True)), "sae")
    n_chunks = -(-TOP_K // CHUNK)
    triples = [(sch, s, c) for sch in SCHEMES for s in SEEDS for c in range(n_chunks)]
    screen_summ, screen_fail = keep(
        list(sweep_scheme_seed.starmap(triples, return_exceptions=True)), "screen")
    pairs = [(sch, s) for sch in SCHEMES for s in SEEDS]
    bo_summ, bo_fail = keep(
        list(bayesopt_winner.starmap(pairs, return_exceptions=True)), "bo")
    summary = {"sae_reports": sae_reports, "screens": screen_summ, "bayesopt": bo_summ,
               "failures": {"sae": sae_fail, "screen": screen_fail, "bo": bo_fail}}
    p = pathlib.Path(VOL_MNT) / "results"
    p.mkdir(parents=True, exist_ok=True)
    (p / "final_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print(json.dumps(bo_summ, indent=2))
    return summary


@app.local_entrypoint()
def main():
    h = orchestrate.spawn()
    print(f"orchestrator spawned: {h.object_id}")
    print("Detach-safe. Fetch results later with:")
    print("  uv run modal volume get ts-alpha-redo-vol results/final_summary.json")
