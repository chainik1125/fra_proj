"""Wide-alpha single-feature steering redo on the randpos-K8 multitrigger sleeper, on Modal.

Equivalent protocol to the TinyStories redo (cloud/modal_alpha_redo.py in the
sae-scaling-sweep worktree), ported to this campaign's own machinery (mts_lib +
run_steer.py conventions, verified against source 2026-07-08):

  Schemes (all layer 0):
    ov        = FRA: ov_diff-ranked ln1-SAE feature, routed through hook_v only
    conventional = act_diff-ranked (sleeper-vs-base) resid_mid-SAE feature, additive
    conv_ln1  = activation-ranked (trigger-span firing) ln1-SAE feature, additive
  Stage 1 screen: top-25 features/scheme x c in [-10,10] step 0.25, PER=8 pairs
    per trigger (64 randpos pairs), greedy 16-token rollouts, J_clean in nats.
  Stage 2: BO (gp_minimize, c in [-20,20]) on the ASR<=0.05-gated winner at
    PER=24, then a HELD-OUT re-measure of c* on fresh prompt rows + positions.

  Splits (all disjoint): eval rows skip=20000, ranking rows skip=22600,
  held-out rows skip=24000, SAE harvest skip=26000.
  Footprint = "prompt" (no trigger-position oracle; rollout feels the edit
  through attention), c>0 subtracts (c=1 = exact feature ablation).

SAEs: canonical campaign recipe (train_saes.py: d2048 k32 s12000 r100000),
seeds {1,2,3,4,5,7}; seed-7 library checkpoints reused from HF when present.

Launch:  modal run --detach cloud/modal_k8_randpos_redo.py    (from this dir)
Fetch:   modal volume get k8-randpos-redo-vol results/final_summary.json
"""
from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
app = modal.App("k8-randpos-redo")
vol = modal.Volume.from_name("k8-randpos-redo-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.57.6",
        "transformer-lens==2.18.0",
        "peft==0.19.1",
        "datasets==4.8.4",
        "typeguard==4.5.1",
        "jaxtyping==0.3.9",
        "einops==0.8.2",
        "accelerate",
        "huggingface_hub",
        "scikit-optimize==0.10.2",
        "numpy",
    )
    .add_local_file(str(HERE / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(HERE / "sae_models.py"), "/work/sae_models.py")
)
secrets = []
if os.environ.get("HF_TOKEN"):
    secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]

VOL = "/vol"
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
PFX = "mts_singlefeat"
ADAPTER = f"{PFX}/artifacts/adapters/randpos_K8"
SEEDS = [1, 2, 7]  # trimmed from {1..5,7} per cost decision 2026-07-08
SCHEMES = ("ov", "conventional", "conv_ln1")
COEFFS = [round(-10.0 + 0.25 * i, 2) for i in range(81)]
TOP_K, CHUNK = 25, 5
SEED = 7
PMIN, PMAX, POOL = 1, 30, 6
SEQ_LEN, MAX_PROMPT, N_NEW = 110, 64, 16
EVAL_SKIP, RANK_SKIP, HELD_SKIP, HARV_SKIP = 20000, 22600, 24000, 26000
PER_SCREEN, PER_FULL = 8, 24
ASR_BAR = 0.05
LN1 = "blocks.0.ln1.hook_normalized"
RMID = "blocks.0.hook_resid_mid"
HOOK_V = "blocks.0.attn.hook_v"
D_SAE, K_SAE, SAE_STEPS, SAE_HARV = 2048, 32, 12000, 100000
SCHEME_HOOK = {"ov": LN1, "conventional": RMID, "conv_ln1": LN1}
SCHEME_SAE = {"ov": "ln1", "conventional": "rmid", "conv_ln1": "ln1"}


def _setup():
    import sys

    sys.path.insert(0, "/work")


def _load_models(device):
    _setup()
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mts_lib as L

    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    tok.pad_token = tok.eos_token
    root = snapshot_download(HF_REPO, repo_type="dataset",
                             allow_patterns=[ADAPTER + "/*"],
                             token=os.environ.get("HF_TOKEN"))
    merged = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),
        str(pathlib.Path(root) / ADAPTER)).merge_and_unload().cpu()
    model = HookedTransformer.from_pretrained(
        L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=device)
    model.eval()
    base_hf = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
    base = HookedTransformer.from_pretrained(
        L.BASE_MODEL, hf_model=base_hf, tokenizer=tok, device=device)
    base.eval()
    return model, base, tok


def _insert_at(clean, ids, p):
    p = max(1, min(p, len(clean)))
    return clean[:p] + list(ids) + clean[p:], p


def _build_pairs(eval_rows, triggers, trigs, per, rng):
    """{trigger: (pairs, groups-by-(len,trig_pos))} with matched clean prompts."""
    from collections import defaultdict

    pbt = {}
    for ti, tn in enumerate(trigs):
        ids, w = triggers[tn]["ids"], triggers[tn]["w"]
        n = len(eval_rows)
        ppool = [rng.randint(PMIN, PMAX) for _ in range(POOL)]
        out, used, j, off = [], 0, 0, ti * per * 2
        while used < per and j < n * 4:
            clean = list(eval_rows[(j + off) % n]["prompt"])
            j += 1
            preq = ppool[used % POOL]
            if len(clean) < preq:
                continue
            dep, p = _insert_at(clean, ids, preq)
            out.append({"clean": clean, "deploy": dep,
                        "trig_pos": list(range(p, p + w))})
            used += 1
        grp = defaultdict(list)
        for i, pr in enumerate(out):
            grp[(len(pr["deploy"]), tuple(pr["trig_pos"]))].append(i)
        pbt[tn] = (out, dict(grp))
    return pbt


def _gen(model, prompts, hooks, device):
    import torch

    with torch.no_grad():
        t = torch.tensor(prompts, device=device)
        P = t.shape[1]
        step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            step.append(lg[:, -1])
            t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)


def _clean_refs(model, pbt, device):
    cc = {}
    for tn, (pairs, grp) in pbt.items():
        for gk, idxs in grp.items():
            _, clog = _gen(model, [pairs[i]["clean"] for i in idxs], [], device)
            cc[(tn, gk)] = clog
    return cc


def _hooks_for(scheme, sae, feats, c, Lp, W_V0, d_model, device):
    """footprint='prompt': patch the first Lp positions only (run_steer.py:317-352)."""
    import torch

    if c == 0 or not feats:
        return []
    ft = torch.tensor(sorted(feats), device=device, dtype=torch.long)
    Wd = sae.W_dec
    read_hook = SCHEME_HOOK[scheme]
    if scheme != "ov":
        def h(x, hook):
            P = x.shape[1]
            z = sae.encode(x.reshape(-1, d_model))
            contrib = (z[:, ft] @ Wd[ft]).reshape(x.shape)
            m = min(Lp, P)
            x[:, :m] = x[:, :m] - c * contrib[:, :m]
            return x
        return [(read_hook, h)]
    cap = {}

    def ln1h(x, hook):
        cap["a"] = x.float()
        return x

    def vh(v, hook):
        a = cap["a"]
        P = a.shape[1]
        z = sae.encode(a.reshape(-1, d_model)).reshape(a.shape[0], P, sae.d_sae)
        z2 = z.clone()
        z2[:, :, ft] = 0.0
        delta = (sae.decode(z2.reshape(-1, sae.d_sae))
                 - sae.decode(z.reshape(-1, sae.d_sae))).reshape(a.shape)
        kd = c * torch.einsum("bpd,hde->bphe", delta, W_V0)
        m = min(Lp, P)
        v[:, :m] = v[:, :m] + kd[:, :m]
        return v
    return [(read_hook, ln1h), (HOOK_V, vh)]


def _evl(model, tok, scheme, sae, feats, c, pbt, cc, W_V0, d_model, device):
    import math

    _setup()
    import mts_lib as L

    asr_sum, n, jcl = 0.0, 0, 0.0
    for tn, (pairs, grp) in pbt.items():
        for gk, idxs in grp.items():
            dp = [pairs[i]["deploy"] for i in idxs]
            hooks = _hooks_for(scheme, sae, feats, c, len(dp[0]), W_V0, d_model, device)
            g, dlog = _gen(model, dp, hooks, device)
            asr_sum += L.asr_from_tokens(g, tok) * len(idxs)
            jcl += L.jsd_rows(dlog, cc[(tn, gk)]).mean(1).sum().item()
            n += len(idxs)
    return {"ASR": round(asr_sum / n, 4), "Jclean_nats": round(jcl / n, 4),
            "Jclean_bits": round(jcl / n / math.log(2), 4)}


def _rank(scheme, model, base, sae, pbt_rank, W_V0, d_model, device, trigs, ihy_probe):
    """Rankings verbatim from run_steer.py:272-309, on DISJOINT ranking pairs."""
    import torch

    with torch.no_grad():
        if scheme == "conventional":  # act_diff: sleeper-vs-base encode diff (all pos)
            acc = torch.zeros(sae.d_sae, device=device)
            cnt = 0
            for tn, (pairs, grp) in pbt_rank.items():
                for gk, idxs in grp.items():
                    dp = torch.tensor([pairs[i]["deploy"] for i in idxs], device=device)
                    _, c1 = model.run_with_cache(dp, return_type=None,
                                                 names_filter=lambda nm: nm == RMID)
                    _, c0 = base.run_with_cache(dp, return_type=None,
                                                names_filter=lambda nm: nm == RMID)
                    acc += (sae.encode(c1[RMID].float().reshape(-1, d_model))
                            - sae.encode(c0[RMID].float().reshape(-1, d_model))).mean(0)
                    cnt += 1
            return torch.argsort(acc / cnt, descending=True).tolist()[:TOP_K]
        pooled = torch.zeros(sae.d_sae, device=device)
        for tn, (pairs, grp) in pbt_rank.items():
            for gk, idxs in grp.items():
                dp = torch.tensor([pairs[i]["deploy"] for i in idxs], device=device)
                _, c1 = model.run_with_cache(dp, return_type=None,
                                             names_filter=lambda nm: nm == LN1)
                a = c1[LN1].float()
                for p in gk[1]:
                    pooled += sae.encode(a[:, p, :]).mean(0)
        cand = set((pooled > 0).nonzero().flatten().tolist())
        if scheme == "conv_ln1":  # activation ranking (trigger-span firing)
            order = torch.argsort(pooled, descending=True).tolist()
            return [f for f in order if f in cand][:TOP_K]
        # ov_diff: |<d_ihy, dW_OV f>| * mean trigger firing, pool-restricted
        W_OV_b = torch.einsum("hde,hef->df", base.W_V[0].float(), base.W_O[0].float())
        W_OV_s = torch.einsum("hde,hef->df", W_V0, model.W_O[0].float())
        dW = (W_OV_s - W_OV_b).detach()
        F = sae.W_dec.detach().float()
        dg = ((F @ dW) @ ihy_probe * (pooled / len(trigs))).detach()
        order = torch.argsort(dg.abs(), descending=True).tolist()
        return [f for f in order if f in cand][:TOP_K]


@app.function(gpu="A10G", image=image, volumes={VOL: vol}, timeout=3 * 3600,
              secrets=secrets, ephemeral_disk=512 * 1024)
def train_saes(hook_kind: str) -> dict:
    """Train/fetch the 6 seed SAEs for one hookpoint (canonical s12000/r100000
    recipe from train_saes.py). Seed-7 library checkpoints reused from HF."""
    _setup()
    import json
    import shutil

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    import mts_lib as L
    from sae_models import TopKSAE

    device = "cuda"
    out_dir = pathlib.Path(VOL) / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    hookname = LN1 if hook_kind == "ln1" else RMID
    saehook = hookname.replace("blocks.0.", "blocks-0-").replace(".", "-") \
        if hook_kind == "ln1" else "hook_resid_mid"
    report = {"hook": hook_kind, "source": {}}

    def path_for(seed):
        return out_dir / f"sae_{hook_kind}_s{seed}.pt"

    missing = [s for s in SEEDS if not path_for(s).exists()]
    # HF library first (canonical schema, dic=sleeper; ln1 also tries dic=base)
    for seed in list(missing):
        for dic in ("sleeper", "union", "base"):
            rel = (f"{PFX}/artifacts/saes_K8_randpos/"
                   f"sae_{dic}_blocks-0-{saehook}_d{D_SAE}_k{K_SAE}"
                   f"_s{SAE_STEPS}_r{SAE_HARV}_seed{seed}.pt")
            try:
                shutil.copy(hf_hub_download(HF_REPO, rel, repo_type="dataset",
                                            token=os.environ.get("HF_TOKEN")),
                            path_for(seed))
                report["source"][seed] = f"hf:{dic}"
                missing.remove(seed)
                break
            except Exception:
                continue
    if missing:
        model, base, tok = _load_models(device)
        d_model = model.cfg.d_model
        pad = tok.eos_token_id
        triggers = L.build_triggers(tok)
        trigs = L.K_SETS[8]
        ihy_ids = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
        import random
        rng = random.Random(SEED + 11)
        harv = L.load_clean_prompts(tok, SAE_HARV, SEQ_LEN, skip=HARV_SKIP,
                                    max_prompt=MAX_PROMPT)

        def padseq(ids):
            ids = ids[:SEQ_LEN]
            m = [1] * len(ids) + [0] * (SEQ_LEN - len(ids))
            return ids + [pad] * (SEQ_LEN - len(ids)), m

        seqs, masks = [], []
        for i, r in enumerate(harv):
            a, m = padseq(r["prompt"] + r["story"])
            seqs.append(a)
            masks.append(m)
            dep, _ = _insert_at(list(r["prompt"]),
                                triggers[trigs[i % len(trigs)]]["ids"],
                                rng.randint(PMIN, PMAX))
            a, m = padseq(dep + ihy_ids)
            seqs.append(a)
            masks.append(m)
        seqs = torch.tensor(seqs)
        masks = torch.tensor(masks).bool()
        total = int(masks.sum())
        mmpath = "/tmp/acts_pool.dat"
        acts = np.memmap(mmpath, dtype=np.float32, mode="w+", shape=(total, d_model))
        off = 0
        with torch.no_grad():
            for s in range(0, seqs.shape[0], 64):
                _, c = model.run_with_cache(seqs[s:s + 64].to(device), return_type=None,
                                            names_filter=lambda n: n == hookname)
                a = c[hookname][masks[s:s + 64].to(device)].float().cpu().numpy()
                acts[off:off + a.shape[0]] = a
                off += a.shape[0]
        acts.flush()
        print(f"[saes {hook_kind}] pool {(total, d_model)} "
              f"~{total * d_model * 4 / 1e9:.1f}GB", flush=True)
        bsum = torch.zeros(d_model, dtype=torch.float64)
        for s in range(0, total, 200000):
            bsum += torch.from_numpy(np.ascontiguousarray(acts[s:s + 200000])).double().sum(0)
        for seed in missing:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            brng = np.random.default_rng(seed)
            sae = TopKSAE(d_in=d_model, d_sae=D_SAE, k=K_SAE).to(device)
            with torch.no_grad():
                sae.b_dec.copy_((bsum / total).float().to(device))
            opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
            for step in range(SAE_STEPS):
                idx = np.sort(brng.integers(0, total, 4096))
                x = torch.from_numpy(np.ascontiguousarray(acts[idx])).to(device).float()
                xh, z = sae(x)
                loss = (x - xh).pow(2).sum(-1).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                with torch.no_grad():
                    sae.normalize_decoder()
            sae.eval()
            with torch.no_grad():
                fvu = float((x - xh).pow(2).sum(-1).mean() / x.pow(2).sum(-1).mean())
            torch.save({"state_dict": sae.state_dict(), "d_in": d_model,
                        "d_sae": D_SAE, "k": K_SAE, "fvu": fvu}, path_for(seed))
            report["source"][seed] = f"trained(fvu={fvu:.4f})"
            vol.commit()
            print(f"[saes {hook_kind}] seed {seed} done fvu={fvu:.4f}", flush=True)
    (out_dir / f"report_{hook_kind}.json").write_text(json.dumps(report, default=str))
    vol.commit()
    print(f"[saes {hook_kind}] {report}", flush=True)
    return report


def _load_sae(hook_kind, seed, device):
    import torch

    from sae_models import TopKSAE

    b = torch.load(pathlib.Path(VOL) / "seeds" / f"sae_{hook_kind}_s{seed}.pt",
                   map_location=device)
    s = TopKSAE(d_in=b["d_in"], d_sae=b["d_sae"], k=b["k"]).to(device)
    s.load_state_dict(b["state_dict"])
    s.eval()
    return s


def _cell_setup(scheme, seed, per, skip, pos_seed):
    """Everything a cell needs: models, sae, pairs+refs, ranking inputs."""
    import random

    import torch

    import mts_lib as L

    device = "cuda"
    model, base, tok = _load_models(device)
    d_model = model.cfg.d_model
    W_V0 = model.W_V[0].float()
    triggers = L.build_triggers(tok)
    trigs = L.K_SETS[8]
    rows = L.load_clean_prompts(tok, 600, SEQ_LEN, skip=skip, max_prompt=MAX_PROMPT)
    pbt = _build_pairs(rows, triggers, trigs, per, random.Random(pos_seed))
    cc = _clean_refs(model, pbt, device)
    sae = _load_sae(SCHEME_SAE[scheme], seed, device)
    # ihy probe: W_U column of the model's first IHY token on a deploy prompt
    dp0 = pbt[trigs[0]][0][0]["deploy"]
    with torch.no_grad():
        id0 = int(model(torch.tensor([dp0], device=device),
                        return_type="logits")[0, -1].argmax())
    ihy_probe = model.W_U[:, id0].detach().float()
    return dict(model=model, base=base, tok=tok, sae=sae, pbt=pbt, cc=cc,
                W_V0=W_V0, d_model=d_model, device=device, trigs=trigs,
                triggers=triggers, ihy_probe=ihy_probe, rows=rows)


@app.function(gpu="A10G", image=image, volumes={VOL: vol}, timeout=7200,
              secrets=secrets)
def sweep_cell(scheme: str, seed: int, chunk: int) -> dict:
    _setup()
    import json
    import random
    import time

    import torch

    p = pathlib.Path(VOL) / "results"
    out_path = p / f"screen_{scheme}_s{seed}_c{chunk}.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        return {"scheme": scheme, "seed": seed, "chunk": chunk,
                "winner": prev["winner"], "cached": True}

    S = _cell_setup(scheme, seed, PER_SCREEN, EVAL_SKIP, SEED + 11)
    # ranking on DISJOINT pairs (skip=22600), per-scheme method
    import mts_lib as L
    rank_rows = L.load_clean_prompts(S["tok"], 300, SEQ_LEN, skip=RANK_SKIP,
                                     max_prompt=MAX_PROMPT)
    pbt_rank = _build_pairs(rank_rows, S["triggers"], S["trigs"], 12,
                            random.Random(SEED + 23))
    ranked = _rank(scheme, S["model"], S["base"], S["sae"], pbt_rank,
                   S["W_V0"], S["d_model"], S["device"], S["trigs"], S["ihy_probe"])
    feats = ranked[chunk * CHUNK:(chunk + 1) * CHUNK]
    print(f"[{scheme} s{seed} c{chunk}] top-{TOP_K}: {ranked} -> chunk {feats}",
          flush=True)

    noint = _evl(S["model"], S["tok"], scheme, S["sae"], [], 0.0, S["pbt"], S["cc"],
                 S["W_V0"], S["d_model"], S["device"])
    rows_out = []
    t0 = time.time()
    with torch.no_grad():
        for fi, f in enumerate(feats):
            for c in COEFFS:
                r = _evl(S["model"], S["tok"], scheme, S["sae"], [f], c, S["pbt"],
                         S["cc"], S["W_V0"], S["d_model"], S["device"])
                rows_out.append({"feat": int(f), "rank": chunk * CHUNK + fi,
                                 "c": c, **r})
            print(f"[{scheme} s{seed} c{chunk}] feat {fi + 1}/{len(feats)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    gated = [r for r in rows_out if r["ASR"] <= ASR_BAR]
    win = min(gated if gated else rows_out,
              key=lambda r: (r["Jclean_nats"], r["ASR"]))
    out = {"scheme": scheme, "seed": seed, "chunk": chunk, "ranked": ranked,
           "chunk_feats": feats, "coeffs": COEFFS, "per_trigger_pairs": PER_SCREEN,
           "no_intervention": noint, "asr_bar": ASR_BAR, "winner": win,
           "rows": rows_out}
    p.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    vol.commit()
    print(f"[{scheme} s{seed} c{chunk}] chunk winner {win}", flush=True)
    return {"scheme": scheme, "seed": seed, "chunk": chunk, "winner": win}


@app.function(gpu="A10G", image=image, volumes={VOL: vol}, timeout=5400,
              secrets=secrets)
def bo_cell(scheme: str, seed: int) -> dict:
    _setup()
    import json

    import torch

    p = pathlib.Path(VOL) / "results"
    bo_path = p / f"bo_{scheme}_s{seed}.json"
    if bo_path.exists():
        prev = json.loads(bo_path.read_text())
        return {k: prev[k] for k in ("scheme", "seed", "feat", "grid_c", "opt_c",
                                     "opt_Jclean_nats", "opt_ASR", "heldout")}
    rows = []
    for c in range(-(-TOP_K // CHUNK)):
        f = p / f"screen_{scheme}_s{seed}_c{c}.json"
        if f.exists():
            rows.extend(json.loads(f.read_text())["rows"])
    if not rows:
        raise RuntimeError(f"no screen chunks for {scheme} s{seed}")
    gated = [r for r in rows if r["ASR"] <= ASR_BAR]
    win = min(gated if gated else rows, key=lambda r: (r["Jclean_nats"], r["ASR"]))
    feat, grid_c = win["feat"], win["c"]

    S = _cell_setup(scheme, seed, PER_FULL, EVAL_SKIP, SEED + 11)
    evals = []

    def objective(x):
        c = float(x[0])
        with torch.no_grad():
            r = _evl(S["model"], S["tok"], scheme, S["sae"], [feat], c, S["pbt"],
                     S["cc"], S["W_V0"], S["d_model"], S["device"])
        loss = r["Jclean_nats"] + 1.0 * max(0.0, r["ASR"] - ASR_BAR)
        evals.append({"c": c, **r, "loss": loss})
        print(f"[bo {scheme} s{seed}] c={c:+.3f} J={r['Jclean_nats']:.4f} "
              f"ASR={r['ASR']:.3f}", flush=True)
        return loss

    from skopt import gp_minimize

    BLO, BHI = -20.0, 20.0
    cand = {grid_c, grid_c - 0.25, grid_c + 0.25, grid_c / 2}
    x0 = [[min(BHI, max(BLO, v))] for v in sorted(cand)]
    gp_minimize(objective, [(BLO, BHI)], x0=x0, n_calls=25, n_initial_points=6,
                random_state=0)
    best = min(evals, key=lambda e: e["loss"])

    # held-out re-measure at c*: fresh rows (skip=24000) + fresh position pool
    H = _cell_setup(scheme, seed, PER_FULL, HELD_SKIP, SEED + 37)
    with torch.no_grad():
        held = _evl(H["model"], H["tok"], scheme, H["sae"], [feat], best["c"],
                    H["pbt"], H["cc"], H["W_V0"], H["d_model"], H["device"])
    out = {"scheme": scheme, "seed": seed, "feat": feat, "grid_c": grid_c,
           "grid_Jclean_screen": win["Jclean_nats"], "opt_c": best["c"],
           "opt_Jclean_nats": best["Jclean_nats"], "opt_Jclean_bits": best["Jclean_bits"],
           "opt_ASR": best["ASR"], "heldout": held, "evals": evals}
    bo_path.write_text(json.dumps(out))
    vol.commit()
    print(f"[bo {scheme} s{seed}] OPT c={best['c']:+.3f} "
          f"J={best['Jclean_nats']:.4f} held={held}", flush=True)
    return {k: out[k] for k in ("scheme", "seed", "feat", "grid_c", "opt_c",
                                "opt_Jclean_nats", "opt_ASR", "heldout")}


@app.function(cpu=2, image=image, volumes={VOL: vol}, timeout=8 * 3600,
              secrets=secrets)
def orchestrate() -> dict:
    import json

    def keep(results, label):
        ok, failed = [], []
        for r in results:
            (failed if isinstance(r, Exception) else ok).append(r)
        for e in failed:
            print(f"[orchestrate] {label} failure: {e!r}", flush=True)
        return ok, [repr(e) for e in failed]

    sae_rep, sae_fail = keep(
        list(train_saes.map(["ln1", "rmid"], return_exceptions=True)), "sae")
    n_chunks = -(-TOP_K // CHUNK)
    triples = [(sch, s, c) for sch in SCHEMES for s in SEEDS for c in range(n_chunks)]
    scr, scr_fail = keep(
        list(sweep_cell.starmap(triples, return_exceptions=True)), "screen")
    pairs = [(sch, s) for sch in SCHEMES for s in SEEDS]
    bo, bo_fail = keep(list(bo_cell.starmap(pairs, return_exceptions=True)), "bo")
    summary = {"sae": sae_rep, "screens": scr, "bo": bo,
               "failures": {"sae": sae_fail, "screen": scr_fail, "bo": bo_fail}}
    p = pathlib.Path(VOL) / "results"
    p.mkdir(parents=True, exist_ok=True)
    (p / "final_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print(json.dumps(bo, indent=2))
    return summary


@app.local_entrypoint()
def main():
    h = orchestrate.spawn()
    print(f"orchestrator spawned: {h.object_id}")
    print("Fetch: modal volume get k8-randpos-redo-vol results/final_summary.json")
