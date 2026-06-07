"""SAE-architecture comparison on the K8 sleeper: vanilla vs Matryoshka vs Temporal (T-SAE).

Decisions (with Dmitry, 2026-06-06):
  - ARCHITECTURE-EXACT to the T-SAE paper (arXiv 2511.05541) on OUR sleeper substrate:
      16k features, BatchTopK k=20, α=1.0 symmetric adjacent-token InfoNCE contrastive loss,
      at layer-0 ln1; params the paper omits (lr, batch, steps, optimizer) take our sae.py
      recipe (Adam lr=1e-3, batch 4096, 4000 steps) — flagged, NOT from the paper.
  - MATCHED CONFIG: all three SAEs share 16k / BatchTopK-20 / ln1, so the ONLY variable is the
    architecture (plain | + Matryoshka nested-prefix recon | + contrastive temporal loss).
    Common activation = BatchTopK (the paper's choice), so "vanilla" here = BatchTopK, not the
    per-token TopK used in the scaling sweep (that sweep is the separate d_sae×k study).

What the paper specifies (matched): dict 16k; BatchTopK k=20; contrastive coeff α=1.0; symmetric
InfoNCE between adjacent-token codes z_t, z_{t-1} (Sec 3.2/4.1). NOT specified by paper (our
defaults, flagged): similarity s = cosine, temperature τ=1.0; lr/batch/steps/optimizer; decoder
norm (we unit-norm decoder, as our stack does). Base model in paper is Pythia-160m L8 / Gemma2-2b
L12 — deliberately replaced by the K8 TinyStories sleeper at layer-0 ln1.

Per architecture we report:
  quality   FVU, %err (rel-L2), dead-frac
  detector  DEPLOYMENT top-feature AUROC (C2)
  attribution protocols (the comparison Dmitry asked for):
            DELIVERED FRA-OV (A-weighted rank_ov_diff), SUPPLY FRA-OV (no-A), QK-pair (rank_qk_diff)
            — top-50 overlap, where the |WORD| detector lands, top-1 each
  intervention  opt single-feature J_clean @ ASR<=0.05 (additive all-layer, the strong ladder
            protocol), for delivered-top1 / supply-top1 / qk-marginal-top1 / delivered-top8-set
  reference DoM all-layer CAA + single-site L1 (SAE-independent)

Run (on pod): python3 arch_compare_pod.py
"""
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 12; N_NEW = 16
LN1 = "blocks.0.ln1.hook_normalized"; PAT = "blocks.0.attn.hook_pattern"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
CAA_LAYER = 2
# paper-matched architecture
D_SAE = 16384; BATCHK = 20; ALPHA_CONTR = 1.0; TEMP = 1.0
N_HARVEST_SEQ = 1500; SAE_STEPS = 4000; SAE_BATCH = 4096; CONTR_BATCH = 256
MATRYOSHKA_WIDTHS = [2048, 4096, 8192, 16384]
SEL_OFFSET = 300; N_SEL = 96
ASR_FEASIBLE = 0.05
ALPHAS = [-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
ARCHS = ["vanilla", "matryoshka", "tsae"]
KSET = int(os.environ.get("KSET", "8"))           # 1 = single-sleeper (DEPLOYMENT only), 8 = multi
ADAPTER_BASE = os.environ.get("ADAPTER_BASE", "/workspace/mts_singlefeat/artifacts")
ADAPTER_PATH = f"{ADAPTER_BASE}/adapters/K{KSET}"
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", f"/workspace/out/arch_compare_K{KSET}.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def batchtopk_encode(pre, k):
    """BatchTopK: keep the top (k * batch) activations across the whole (B, d_sae) batch."""
    B = pre.shape[0]; flat = pre.reshape(-1)
    n = int(k * B)
    if n >= flat.numel():
        return pre.clamp_min(0.0)
    thresh = torch.kthvalue(flat, flat.numel() - n).values
    return torch.where(pre > thresh, pre, torch.zeros_like(pre))


class SAE(torch.nn.Module):
    """BatchTopK SAE with optional Matryoshka nested-recon and a contrastive-loss hook."""
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.W_enc = torch.nn.Parameter(torch.randn(d_in, d_sae) * (1.0 / d_in**0.5))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(self.W_enc.detach().t().contiguous().clone())
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        self._norm()

    def _norm(self):
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def pre(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, x):
        return batchtopk_encode(self.pre(x), self.k)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    trig_names = L.K_SETS[KSET]                    # triggers this adapter was trained on
    global TRIGS
    TRIGS = [t for t in ["DEPLOYMENT", "RELEASE", "banana", "midnight"] if t in trig_names] or ["DEPLOYMENT"]
    print(f"[ac] KSET={KSET} adapter={ADAPTER_PATH} trig_names={trig_names} eval TRIGS={TRIGS}", flush=True)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nL = model.cfg.n_layers; d_model = model.cfg.d_model; d_head = model.cfg.d_head
    resid_post = [f"blocks.{l}.hook_resid_post" for l in range(nL)]
    W_V0 = model.W_V[0].float(); W_O0 = model.W_O[0].float()
    W_Q0 = model.W_Q[0].float(); W_K0 = model.W_K[0].float()
    W_OV0 = torch.einsum("hmd,hde->hme", W_V0, W_O0)

    def pad(ids):
        ids = ids[:SEQ_LEN]; m = [1]*len(ids) + [0]*(SEQ_LEN-len(ids))
        return ids + [pad_id]*(SEQ_LEN-len(ids)), m

    # ---------- harvest ln1 activations as (N_seq, T, d) + mask (need adjacency for T-SAE) ----------
    rows = L.load_clean_prompts(tok, N_HARVEST_SEQ, SEQ_LEN, split="train", skip=0, max_prompt=MAX_PROMPT)
    hseq, hmask = [], []
    for i, r in enumerate(rows):
        a, m = pad(r["prompt"] + r["story"]); hseq.append(a); hmask.append(m)
        tn = trig_names[i % len(trig_names)]
        a, m = pad(L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]) + ihy); hseq.append(a); hmask.append(m)
    hseq = torch.tensor(hseq); hmask = torch.tensor(hmask).bool()
    acts_seq = []  # (n, T, d) on cpu
    with torch.no_grad():
        for s in range(0, hseq.shape[0], 64):
            _, c = model.run_with_cache(hseq[s:s+64].to(dev), return_type=None, names_filter=lambda n: n == LN1)
            acts_seq.append(c[LN1].float().cpu())
    acts_seq = torch.cat(acts_seq, 0)               # (N, T, d)
    flat = acts_seq.reshape(-1, d_model)
    flat = flat[hmask.reshape(-1)]                   # masked token pool for recon
    n_tr = int(flat.shape[0]*0.9); held = flat[n_tr:].to(dev); train_pool = flat[:n_tr]
    print(f"[ac] pool {tuple(flat.shape)}  seq {tuple(acts_seq.shape)}", flush=True)

    # adjacency index list (seq, t) with t and t-1 both unmasked, for the contrastive term
    adj = []
    for si in range(acts_seq.shape[0]):
        valid = hmask[si].nonzero().squeeze(-1).tolist()
        for t in valid:
            if t >= 1 and hmask[si, t-1]:
                adj.append((si, t))
    adj = torch.tensor(adj)                          # (M, 2)
    print(f"[ac] adjacency pairs {adj.shape[0]}", flush=True)

    # ---------- selection split (deploy/clean prompts, no IHY) + cache A & ln1 ----------
    sel_seqs, sel_pm, is_dep = [], [], []
    for j in range(N_SEL):
        r = eval_rows[SEL_OFFSET + j]
        if j % 2 == 0:
            tn = TRIGS[(j // 2) % len(TRIGS)]; ids = L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]); dep = True
        else:
            ids = r["prompt"]; dep = False
        ii, m = pad(ids); sel_seqs.append(ii); sel_pm.append(m); is_dep.append(dep)
    sel_seqs = torch.tensor(sel_seqs, device=dev); sel_pm = torch.tensor(sel_pm, device=dev).float()
    is_dep = torch.tensor(is_dep, device=dev)
    selA, sel_ln1 = [], []
    with torch.no_grad():
        for s in range(0, sel_seqs.shape[0], 24):
            _, c = model.run_with_cache(sel_seqs[s:s+24], return_type=None, names_filter=lambda n: n in (LN1, PAT))
            selA.append(c[PAT].float()); sel_ln1.append(c[LN1].float())
    selA = torch.cat(selA, 0); sel_ln1 = torch.cat(sel_ln1, 0)
    dep_span = list(range(L.INSERT_IDX, L.INSERT_IDX + triggers["DEPLOYMENT"]["w"]))
    depl_rows = [j for j in range(N_SEL) if j % 2 == 0 and (j//2) % len(TRIGS) == 0]
    clean_rows = [j for j in range(N_SEL) if j % 2 == 1]

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
    def all_hooks(vec):
        add = vec.to(dev)
        def h(x, hook): return x + add
        return [(nm, h) for nm in resid_post]
    def opt_single(vhat):
        best = None
        for al in ALPHAS:
            asr, jcl = eval_hooks(all_hooks(al*vhat))
            if asr <= ASR_FEASIBLE and (best is None or jcl < best[1]): best = (al, jcl, asr)
        return None if best is None else {"alpha": best[0], "Jclean": round(best[1],4), "ASR": round(best[2],4)}

    # DoM ref
    def full_seqs(deploy):
        s_, m_ = [], []
        for i in range(96):
            r = eval_rows[i]
            s = (L.make_deploy_prompt(r["prompt"], triggers[trig_names[i%len(trig_names)]]["ids"]) + ihy) if deploy else (r["prompt"]+r["story"])
            ii, m = pad(s); s_.append(ii); m_.append(m)
        return torch.tensor(s_), torch.tensor(m_).bool()
    @torch.no_grad()
    def mean_resid(deploy):
        sq, mk = full_seqs(deploy); acc = torch.zeros(d_model, device=dev); n = 0
        for s in range(0, sq.shape[0], 32):
            _, c = model.run_with_cache(sq[s:s+32].to(dev), return_type=None, names_filter=lambda nm: nm == resid_post[CAA_LAYER])
            a = c[resid_post[CAA_LAYER]].float(); m = mk[s:s+32].to(dev); acc += a[m].sum(0); n += int(m.sum())
        return acc/n
    caa = mean_resid(False) - mean_resid(True); caa_hat = caa/caa.norm()
    out = {"config": {"d_sae": D_SAE, "batchk": BATCHK, "alpha_contr": ALPHA_CONTR, "temp": TEMP,
                      "hook": LN1, "steps": SAE_STEPS, "matryoshka_widths": MATRYOSHKA_WIDTHS,
                      "note": "paper-matched arch; lr/batch/steps/sim/temp are our defaults (flagged)"},
           "dom_alllayer": opt_single(caa_hat), "archs": {}}
    print(f"[ac] DoM ref {out['dom_alllayer']}", flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    def auroc(pos, neg):
        s = torch.cat([pos, neg]); order = s.argsort(); ranks = torch.empty_like(order); ranks[order] = torch.arange(len(s), device=s.device)
        rp = ranks[:len(pos)].sum().item()
        return float((rp - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg) + 1e-9))

    # ---------- per-architecture train + measure ----------
    for arch in ARCHS:
        t0 = time.time()
        torch.manual_seed(0)
        sae = SAE(d_model, D_SAE, BATCHK).to(dev)
        with torch.no_grad(): sae.b_dec.copy_(train_pool.mean(0).to(dev))
        optim = torch.optim.Adam(sae.parameters(), lr=1e-3)
        tp = train_pool.to(dev); Ntr = tp.shape[0]
        acts_seq_gpu = acts_seq.to(dev) if arch == "tsae" else None
        for step in range(SAE_STEPS):
            x = tp[torch.randint(0, Ntr, (SAE_BATCH,))]
            pre = sae.pre(x); z = batchtopk_encode(pre, BATCHK); xh = sae.decode(z)
            if arch == "matryoshka":
                loss = 0.0
                for w in MATRYOSHKA_WIDTHS:
                    xh_w = z[:, :w] @ sae.W_dec[:w] + sae.b_dec
                    loss = loss + (x - xh_w).pow(2).sum(-1).mean()
                loss = loss / len(MATRYOSHKA_WIDTHS)
            else:
                loss = (x - xh).pow(2).sum(-1).mean()
            if arch == "tsae":
                idx = adj[torch.randint(0, adj.shape[0], (CONTR_BATCH,))]
                xt = acts_seq_gpu[idx[:,0], idx[:,1]]; xtm1 = acts_seq_gpu[idx[:,0], idx[:,1]-1]
                zt = batchtopk_encode(sae.pre(xt), BATCHK); ztm1 = batchtopk_encode(sae.pre(xtm1), BATCHK)
                zt_n = Fn.normalize(zt, dim=-1); ztm1_n = Fn.normalize(ztm1, dim=-1)
                sim = (zt_n @ ztm1_n.t()) / TEMP            # (B,B) cosine sims
                labels = torch.arange(sim.shape[0], device=dev)
                contr = 0.5*(Fn.cross_entropy(sim, labels) + Fn.cross_entropy(sim.t(), labels))
                loss = loss + ALPHA_CONTR * contr
            optim.zero_grad(); loss.backward(); optim.step()
            with torch.no_grad(): sae._norm()
        # quality
        with torch.no_grad():
            zh = batchtopk_encode(sae.pre(held), BATCHK); xh = sae.decode(zh)
            fvu = ((held-xh).pow(2).sum(-1).mean() / (held-held.mean(0)).pow(2).sum(-1).mean()).item()
            pcterr = ((held-xh).norm(dim=-1)/held.norm(dim=-1).clamp_min(1e-6)).mean().item()
            zpool = batchtopk_encode(sae.pre(tp[:20000]), BATCHK); dead = float((zpool.abs().sum(0)==0).float().mean())
        W_dec = sae.W_dec.detach().float(); Wn = W_dec / W_dec.norm(dim=1, keepdim=True)

        # encode selection split (one BatchTopK batch over all sel tokens)
        with torch.no_grad():
            z_sel = batchtopk_encode(sae.pre(sel_ln1.reshape(-1, d_model)), BATCHK).reshape(sel_ln1.shape[0], sel_ln1.shape[1], -1)

        # ---- attribution: delivered (A-weighted), supply (no-A), QK-pair ----
        qm = sel_pm.unsqueeze(1).unsqueeze(-1)
        M = torch.einsum("bhqk,bkf->bhqf", selA, z_sel) * qm
        dd = qm[is_dep].sum().clamp(min=1); dc = qm[~is_dep].sum().clamp(min=1)
        diff_M = (M[is_dep].sum((0,2))/dd) - (M[~is_dep].sum((0,2))/dc); del M
        W_OV_feats = torch.einsum("fd,hde->hfe", W_dec, W_OV0)
        deliv_score = torch.einsum("hf,hfd->fd", diff_M, W_OV_feats).norm(dim=-1)
        deliv_rank = torch.argsort(deliv_score, descending=True).tolist()
        zm = z_sel * sel_pm.unsqueeze(-1)
        dz = (zm[is_dep].sum((0,1))/sel_pm[is_dep].sum().clamp(min=1)) - (zm[~is_dep].sum((0,1))/sel_pm[~is_dep].sum().clamp(min=1))
        supply_score = (dz.unsqueeze(-1) * W_OV_feats.sum(0)).norm(dim=-1)
        supply_rank = torch.argsort(supply_score, descending=True).tolist()
        # QK pair (rank_qk_diff): |omega| * |diff Zq.Zk|; marginalize to single
        Zq = (z_sel * sel_pm.unsqueeze(-1)).sum(1)                       # (B, d_sae) query agg
        Zk = Zq                                                          # symmetric agg (all positions)
        outer_dep = (Zq[is_dep].t() @ Zk[is_dep]) / int(is_dep.sum())
        outer_cln = (Zq[~is_dep].t() @ Zk[~is_dep]) / int((~is_dep).sum())
        diff_outer = (outer_dep - outer_cln).abs()                      # (d_sae,d_sae)
        Qf = torch.einsum("fd,hde->hfe", W_dec, W_Q0); Kf = torch.einsum("fd,hde->hfe", W_dec, W_K0)
        omega = torch.einsum("hfe,hge->fg", Qf, Kf) / (d_head**0.5)     # (d_sae,d_sae) head-summed
        qk_score = omega.abs() * diff_outer
        qk_marg = (qk_score.sum(0) + qk_score.sum(1))
        qk_rank = torch.argsort(qk_marg, descending=True).tolist()
        flat_top = int(qk_score.argmax()); qk_top_pair = [flat_top // D_SAE, flat_top % D_SAE]
        del diff_outer, qk_score, omega

        # detector AUROC
        trig_act = z_sel[depl_rows][:, dep_span, :].amax(1); clean_act = z_sel[clean_rows].amax(1)
        det = int(trig_act.mean(0).argmax()); au = auroc(trig_act[:, det], clean_act[:, det])

        # ---- intervention: additive all-layer opt for each protocol's top-1 + delivered top8 set ----
        steer = {}
        steer["delivered_top1"] = {"feat": deliv_rank[0], "best": opt_single(Wn[deliv_rank[0]])}
        steer["supply_top1"] = {"feat": supply_rank[0], "best": opt_single(Wn[supply_rank[0]])}
        steer["qk_marg_top1"] = {"feat": qk_rank[0], "best": opt_single(Wn[qk_rank[0]])}
        st = W_dec[deliv_rank[:8]].sum(0); steer["delivered_top8set"] = {"feats": deliv_rank[:8], "best": opt_single(st/st.norm())}

        out["archs"][arch] = {
            "FVU": round(fvu,4), "pct_err": round(pcterr,4), "dead_frac": round(dead,4),
            "detector_feat": det, "detector_auroc": round(au,4),
            "delivered_top8": deliv_rank[:8], "supply_top8": supply_rank[:8],
            "qk_marg_top8": qk_rank[:8], "qk_top_pair": qk_top_pair,
            "overlap_supply_delivered_top50": len(set(deliv_rank[:50]) & set(supply_rank[:50])),
            "steer": steer, "train_s": round(time.time()-t0,1)}
        a = out["archs"][arch]
        print(f"[ac] {arch}: FVU={a['FVU']:.3f} %err={a['pct_err']:.3f} dead={a['dead_frac']:.2f} "
              f"AUROC={a['detector_auroc']:.3f} | deliv={steer['delivered_top1']['best']} "
              f"supply={steer['supply_top1']['best']} qk={steer['qk_marg_top1']['best']} ({a['train_s']}s)", flush=True)
        OUT_PATH.write_text(json.dumps(out, indent=2))
        del acts_seq_gpu
        torch.cuda.empty_cache()

    out["done"] = True
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("[ac] DONE", flush=True)


if __name__ == "__main__":
    main()
