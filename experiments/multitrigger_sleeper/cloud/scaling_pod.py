"""Multi-sleeper SAE-scaling measurement — the K8 equivalent of sae_scaling/RESULTS.md.

Mirrors the May-25 single-sleeper scaling study, on the K8 multi-trigger model:
  grid   d_sae ∈ {1536,3072,6144} × k ∈ {10,32,50} × hookpoint ∈ {ln1, resid_mid}
  per pod: ONE seed (run 3 pods, one per seed — matches the original's one-worker-per-seed).
  metrics per checkpoint:
    SAE quality   FVU (held-out), %err (rel-L2), dead-frac
    steering      opt single-feature J_clean @ ASR<=0.05 over a signed α-grid, in each cell's
                  NATIVE protocol (faithful to the original):
                    ln1 / "OV/OV"      : FRA-OV-DELIVERED ranking (rank_ov_diff, validated
                                         2026-06-06), steer top-1 via V-channel (hook_v, layer 0)
                    resid_mid / "conv" : dep−clean activation-diff ranking, steer top-1 additively
                                         at blocks.0.hook_resid_mid
    detector      DEPLOYMENT top-feature AUROC (C2 isolation, ~free)
  reference (SAE-independent, computed once): DoM all-layer CAA (≈0.269) + single-site L1 (≈0.247).

New-vs-original framing: the original asked "does SAE config affect steerability?" (answer:
width/k matter, steps flat, rec⊥steer). Here the payload is DISTRIBUTED across triggers/features,
so the live question is whether ANY (d_sae,k) lifts a single feature toward the DoM line — and
whether feature-splitting at higher width makes detection/steering better or worse.

Run (on pod): SEED=0 python3 scaling_pod.py   (needs: pip install scikit-optimize not required)
"""
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 12; N_NEW = 16
LN1 = "blocks.0.ln1.hook_normalized"; PAT = "blocks.0.attn.hook_pattern"; RMID = "blocks.0.hook_resid_mid"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
CAA_LAYER = 2
N_HARVEST_ROWS = 1500; SAE_STEPS = 4000; SAE_BATCH = 4096
SEL_OFFSET = 300; N_SEL = 96
D_SAES = [1536, 3072, 6144]; KS = [10, 32, 50]
HOOKS = ["ln1", "resid_mid"]
SEED = int(os.environ.get("SEED", "0"))
ASR_FEASIBLE = 0.05
ALPHAS = [-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
SCREEN = [-16.0, -8.0, -4.0, 4.0, 8.0, 16.0]   # coarse per-feature screen for the top-N search
TOP_N = 20                                       # best-of-top-N by attribution (Dmitry)

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", f"/workspace/out/scaling_seed{SEED}.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    trig_names = L.K_SETS[8]
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nL = model.cfg.n_layers; d_model = model.cfg.d_model
    resid_post = [f"blocks.{l}.hook_resid_post" for l in range(nL)]
    W_OV0 = torch.einsum("hmd,hde->hme", model.W_V[0].float(), model.W_O[0].float())   # (h,d,d)
    W_V0 = model.W_V[0].float()

    def pad(ids):
        ids = ids[:SEQ_LEN]; m = [1]*len(ids) + [0]*(SEQ_LEN-len(ids))
        return ids + [pad_id]*(SEQ_LEN-len(ids)), m

    # ---------- harvest pools (once per hookpoint; SAE-independent) ----------
    rows = L.load_clean_prompts(tok, N_HARVEST_ROWS, SEQ_LEN, split="train", skip=0, max_prompt=MAX_PROMPT)
    hseqs, hmask = [], []
    for i, r in enumerate(rows):
        a, m = pad(r["prompt"] + r["story"]); hseqs.append(a); hmask.append(m)
        tn = trig_names[i % len(trig_names)]
        a, m = pad(L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]) + ihy); hseqs.append(a); hmask.append(m)
    hseqs = torch.tensor(hseqs); hmask = torch.tensor(hmask)

    def harvest(hookname):
        acts = []
        with torch.no_grad():
            for s in range(0, hseqs.shape[0], 64):
                b = hseqs[s:s+64].to(dev); bm = hmask[s:s+64].to(dev).bool()
                _, c = model.run_with_cache(b, return_type=None, names_filter=lambda n: n == hookname)
                acts.append(c[hookname][bm].float().cpu())
        return torch.cat(acts, 0)

    pools = {"ln1": harvest(LN1), "resid_mid": harvest(RMID)}
    for hk in pools: print(f"[sc] pool {hk} = {tuple(pools[hk].shape)}", flush=True)

    # ---------- selection split: deploy/clean prompts (no IHY), cache raw acts + A ----------
    sel_seqs, sel_pm, is_dep = [], [], []
    for j in range(N_SEL):
        r = eval_rows[SEL_OFFSET + j]
        if j % 2 == 0:
            tn = TRIGS[(j // 2) % len(TRIGS)]; ids = L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]); dep = True
        else:
            ids = r["prompt"]; dep = False
        ids = ids[:SEQ_LEN]; m = [1]*len(ids) + [0]*(SEQ_LEN-len(ids)); ids = ids + [pad_id]*(SEQ_LEN-len(ids))
        sel_seqs.append(ids); sel_pm.append(m); is_dep.append(dep)
    sel_seqs = torch.tensor(sel_seqs, device=dev); sel_pm = torch.tensor(sel_pm, device=dev).float()
    is_dep = torch.tensor(is_dep, device=dev)
    selA, sel_ln1, sel_rmid = [], [], []
    with torch.no_grad():
        for s in range(0, sel_seqs.shape[0], 24):
            b = sel_seqs[s:s+24]
            _, c = model.run_with_cache(b, return_type=None, names_filter=lambda n: n in (LN1, PAT, RMID))
            selA.append(c[PAT].float()); sel_ln1.append(c[LN1].float()); sel_rmid.append(c[RMID].float())
    selA = torch.cat(selA, 0); sel_ln1 = torch.cat(sel_ln1, 0); sel_rmid = torch.cat(sel_rmid, 0)
    sel_raw = {"ln1": sel_ln1, "resid_mid": sel_rmid}

    # detector-AUROC corpus: DEPLOYMENT trigger span vs clean (reuse selection split)
    dep_span = list(range(L.INSERT_IDX, L.INSERT_IDX + triggers["DEPLOYMENT"]["w"]))

    # ---------- steering harness (shared clean cache) ----------
    @torch.no_grad()
    def greedy_logits(prompts, hooks):
        t = torch.tensor(prompts, device=dev); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    pairs_by_trig = {}; clean_cache = {}
    for tn in TRIGS:
        pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER)
        grp = defaultdict(list)
        for i, p in enumerate(pairs): grp[len(p["clean"])].append(i)
        pairs_by_trig[tn] = (pairs, grp)
        for Lc, idxs in grp.items():
            _, clog = greedy_logits([pairs[i]["clean"] for i in idxs], [])
            clean_cache[(tn, Lc)] = clog
    print("[sc] clean cache built", flush=True)

    def eval_hooks(hooks):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs, grp = pairs_by_trig[tn]
            for Lc, idxs in grp.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                g, dlog = greedy_logits(dp, hooks)
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clean_cache[(tn, Lc)]).mean(1).sum().item(); ntot += len(idxs)
        return asr/ntot, jcl/ntot

    def opt_single(vhat_builder, grid=ALPHAS):
        best = None
        for al in grid:
            asr, jcl = eval_hooks(vhat_builder(al))
            if asr <= ASR_FEASIBLE and (best is None or jcl < best[1]):
                best = (al, jcl, asr)
        return best   # (alpha, Jclean, asr) or None

    def search_topN(ranked, builder_factory):
        """Best-of-top-N by attribution: coarse-screen each of top-N, refine the winner +
        rank-1 over the full grid. Returns (best20 dict, rank1 dict, n_suppress)."""
        screened = []
        for f in ranked[:TOP_N]:
            b = builder_factory(f); bf = opt_single(b, SCREEN)
            screened.append((f, bf))
        n_supp = sum(1 for _, bf in screened if bf is not None)
        feas = [(f, bf) for f, bf in screened if bf is not None]
        def refine(f, seed_best):
            b = builder_factory(f); full = opt_single(b, ALPHAS)
            cands = [c for c in (seed_best, full) if c is not None]
            return min(cands, key=lambda c: c[1]) if cands else None
        best20 = None
        if feas:
            fw, bw = min(feas, key=lambda x: x[1][1]); r = refine(fw, bw)
            if r: best20 = {"feat": fw, "alpha": r[0], "Jclean": round(r[1],4), "ASR": round(r[2],4)}
        f1 = ranked[0]; r1 = refine(f1, dict_get(screened, f1))
        rank1 = None if r1 is None else {"feat": f1, "alpha": r1[0], "Jclean": round(r1[1],4), "ASR": round(r1[2],4)}
        return best20, rank1, n_supp

    def dict_get(screened, f):
        for ff, bf in screened:
            if ff == f: return bf
        return None

    # ---------- DoM reference (SAE-independent, once) ----------
    def full_seqs(deploy):
        s_, m_ = [], []
        for i in range(96):
            r = eval_rows[i]
            s = (L.make_deploy_prompt(r["prompt"], triggers[trig_names[i % 8]]["ids"]) + ihy) if deploy else (r["prompt"] + r["story"])
            s = s[:SEQ_LEN]; m = [1]*len(s) + [0]*(SEQ_LEN-len(s)); s = s + [pad_id]*(SEQ_LEN-len(s))
            s_.append(s); m_.append(m)
        return torch.tensor(s_), torch.tensor(m_).bool()
    @torch.no_grad()
    def mean_resid(deploy):
        sq, mk = full_seqs(deploy); acc = torch.zeros(d_model, device=dev); n = 0
        for s in range(0, sq.shape[0], 32):
            _, c = model.run_with_cache(sq[s:s+32].to(dev), return_type=None, names_filter=lambda nm: nm == resid_post[CAA_LAYER])
            a = c[resid_post[CAA_LAYER]].float(); m = mk[s:s+32].to(dev); acc += a[m].sum(0); n += int(m.sum())
        return acc / n
    caa = mean_resid(False) - mean_resid(True); caa_hat = caa / caa.norm()
    def all_hooks(vec):
        add = vec.to(dev)
        def h(x, hook): return x + add
        return [(nm, h) for nm in resid_post]
    def l1_hooks(vec):
        add = vec.to(dev)
        def h(x, hook): return x + add
        return [(resid_post[1], h)]
    dom_all = opt_single(lambda al: all_hooks(al * caa_hat))
    dom_l1 = opt_single(lambda al: l1_hooks(al * caa_hat))
    print(f"[sc] DoM ref: all-layer {dom_all}  L1 {dom_l1}", flush=True)

    results = {"seed": SEED, "dom_alllayer": dom_all, "dom_L1": dom_l1, "grid": {}}

    def auroc(pos, neg):
        pos = pos.detach(); neg = neg.detach()
        s = torch.cat([pos, neg]); order = s.argsort()
        ranks = torch.empty_like(order); ranks[order] = torch.arange(len(s), device=s.device)
        rp = ranks[:len(pos)].sum().item()
        return float((rp - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg) + 1e-9))

    for hk in HOOKS:
        pool = pools[hk].to(dev)
        n_tr = int(pool.shape[0]*0.9); train_pool = pool[:n_tr]; held = pool[n_tr:]
        for d_sae in D_SAES:
            for k in KS:
                key = f"{hk}_d{d_sae}_k{k}"
                t0 = time.time()
                torch.manual_seed(SEED)
                sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=k).to(dev)
                with torch.no_grad(): sae.b_dec.copy_(train_pool.mean(0))
                opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
                Ntr = train_pool.shape[0]
                for step in range(SAE_STEPS):
                    x = train_pool[torch.randint(0, Ntr, (SAE_BATCH,))]
                    xh, z = sae(x); loss = (x - xh).pow(2).sum(-1).mean()
                    loss.backward(); opt.step(); opt.zero_grad()
                    with torch.no_grad(): sae.normalize_decoder()
                # quality on held-out
                with torch.no_grad():
                    xh, z = sae(held)
                    fvu = ((held-xh).pow(2).sum(-1).mean() / (held-held.mean(0)).pow(2).sum(-1).mean()).item()
                    pcterr = ((held-xh).norm(dim=-1) / held.norm(dim=-1).clamp_min(1e-6)).mean().item()
                    zfull = sae.encode(pool); dead = float((zfull.abs().sum(0) == 0).float().mean())
                W_dec = sae.W_dec.detach().float()

                # ranking + steering, native protocol per cell — best-of-top-N by attribution
                if hk == "ln1":
                    z_sel = sae.encode(sel_ln1.reshape(-1, d_model)).reshape(sel_ln1.shape[0], sel_ln1.shape[1], -1)
                    qm = sel_pm.unsqueeze(1).unsqueeze(-1)
                    M = torch.einsum("bhqk,bkf->bhqf", selA, z_sel) * qm
                    dd = qm[is_dep].sum().clamp(min=1); dc = qm[~is_dep].sum().clamp(min=1)
                    diff_M = (M[is_dep].sum((0,2))/dd) - (M[~is_dep].sum((0,2))/dc)
                    W_OV_feats = torch.einsum("fd,hde->hfe", W_dec, W_OV0)
                    score = torch.einsum("hf,hfd->fd", diff_M, W_OV_feats).norm(dim=-1)
                    Wn = W_dec / W_dec.norm(dim=1, keepdim=True)
                    def builder_factory(f):                 # V-channel (layer-0 hook_v)
                        vd = torch.einsum("d,hde->he", Wn[f], W_V0)
                        def b(al, vd=vd):
                            add = al*vd
                            def h(v, hook): return v + add
                            return [("blocks.0.attn.hook_v", h)]
                        return b
                else:
                    z_sel = sae.encode(sel_rmid.reshape(-1, d_model)).reshape(sel_rmid.shape[0], sel_rmid.shape[1], -1)
                    zm = z_sel * sel_pm.unsqueeze(-1)
                    dd = sel_pm[is_dep].sum().clamp(min=1); dc = sel_pm[~is_dep].sum().clamp(min=1)
                    score = (zm[is_dep].sum((0,1))/dd) - (zm[~is_dep].sum((0,1))/dc)   # dep−clean diff
                    Wn = W_dec / W_dec.norm(dim=1, keepdim=True)
                    def builder_factory(f):                 # additive at resid_mid
                        def b(al, f=f):
                            add = al*Wn[f]
                            def h(x, hook): return x + add
                            return [(RMID, h)]
                        return b
                ranked = torch.argsort(score, descending=True).tolist()
                best20, rank1, n_supp = search_topN(ranked, builder_factory)

                # detector AUROC for DEPLOYMENT top feature
                zsel = sae.encode(sel_raw[hk].reshape(-1, d_model)).reshape(sel_raw[hk].shape[0], sel_raw[hk].shape[1], -1)
                dep_idx = (is_dep).nonzero().squeeze(-1)
                # DEPLOYMENT prompts are the dep entries whose trigger is DEPLOYMENT (j%2==0, (j//2)%4==0)
                depl_rows = [j for j in range(N_SEL) if j % 2 == 0 and (j//2) % len(TRIGS) == 0]
                trig_act = zsel[depl_rows][:, dep_span, :].amax(1)   # (n, d_sae) max over span
                clean_rows = [j for j in range(N_SEL) if j % 2 == 1]
                clean_act = zsel[clean_rows].amax(1)
                topf = int(trig_act.mean(0).argmax())
                au = auroc(trig_act[:, topf], clean_act[:, topf])

                results["grid"][key] = {"hook": hk, "d_sae": d_sae, "k": k,
                    "FVU": round(fvu,4), "pct_err": round(pcterr,4), "dead_frac": round(dead,4),
                    "steer_best20": best20, "steer_rank1": rank1, "n_suppress_of20": n_supp,
                    "detector_feat": topf, "detector_auroc": round(au,4),
                    "train_s": round(time.time()-t0,1)}
                b = results["grid"][key]
                b20 = b["steer_best20"]["Jclean"] if b["steer_best20"] else "FAIL"
                r1 = b["steer_rank1"]["Jclean"] if b["steer_rank1"] else "FAIL"
                print(f"[sc] {key}: FVU={b['FVU']:.3f} %err={b['pct_err']:.3f} dead={b['dead_frac']:.2f} "
                      f"best20J={b20} rank1J={r1} nsupp={n_supp}/20 AUROC={b['detector_auroc']:.3f} ({b['train_s']}s)", flush=True)
                OUT_PATH.write_text(json.dumps(results, indent=2))

    results["done"] = True
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print("[sc] DONE", flush=True)


if __name__ == "__main__":
    main()
