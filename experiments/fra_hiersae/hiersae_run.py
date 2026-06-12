# EXTRA_PIP:
"""HIERSAE x FRA synthetic — does a NAIVE hierarchical-SAE + FRA solve drift?
(PHASE B / EVALUATOR; spec = HIERSAE_DESIGN.md + PREREG.md).

Self-contained: planted-hierarchy generator (extends synth_hier2.py with the
m-leaf/context drift + per-token labels + superpos knob), a flat TopKSAE and a
prefix-Matryoshka SAE (the minimal §2.2 add), a DIRECT 4-D FRA cell
decomposition for the synthetic single-linear-QK head (no RoPE/LN), the §3
higher-hierarchy-cells level-spreading + coarse-cut residual, the §4 three-axis
metrics, the §5 gates, and the locked §5.1 CONFIRM/FALSIFY/HELP verdict.

GROUND TRUTH everywhere (we know the planted hierarchy) — no judge, no API.
CPU, minutes. ckpt() inside the alpha x superpos x seed loop; top-level
try/except uploads the traceback (crash-before-upload guard).
"""
import os, sys, json, time, math, traceback, pathlib

import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)

# ───────────────────────── output / HF plumbing ─────────────────────────
OUT = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/hiersae_results.json"))
OUT.parent.mkdir(parents=True, exist_ok=True)
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
PFX = "fra_hiersae/results"
TOK = os.environ.get("HF_TOKEN")


def _upload(local, repo_path):
    try:
        from huggingface_hub import upload_file
        for _ in range(6):
            try:
                upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                            repo_id=HF_REPO, repo_type="dataset", token=TOK)
                return True
            except Exception as e:
                print(f"[upload retry] {e}", flush=True); time.sleep(20)
    except Exception as e:
        print(f"[upload unavailable] {e}", flush=True)
    return False


# ═══════════════════════════ 1. SAE classes ═════════════════════════════
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k, use_relu=True):
        super().__init__()
        self.d_in, self.d_sae, self.k, self.use_relu = d_in, d_sae, k, use_relu
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.use_relu:
            pre = torch.relu(pre)
        tv, ti = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, ti, tv)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def compute_loss(self, x):
        xh, z = self.forward(x)
        rl = (x - xh).pow(2).sum(-1).mean()
        return rl, {"full_recon": rl.item()}

    @torch.no_grad()
    def normalize_decoder(self):
        self._normalize_decoder()


class MatryoshkaSAE(TopKSAE):
    """Plain prefix-nested TopK SAE (single-token analogue of the temporal
    Matryoshka crosscoder). Coarse = [0:w0], fine = the rest."""
    def __init__(self, d_in, d_sae, k, matryoshka_widths, inner_weight=1.0, use_relu=True):
        super().__init__(d_in, d_sae, k, use_relu)
        self.matryoshka_widths = sorted(set(list(matryoshka_widths) + [d_sae]))
        self.inner_weight = inner_weight

    def _decode_prefix(self, z, w):
        return z[:, :w] @ self.W_dec[:w, :] + self.b_dec

    def compute_loss(self, x):
        z = self.encode(x)
        full = (x - self.decode(z)).pow(2).sum(-1).mean()
        inner = [(x - self._decode_prefix(z, w)).pow(2).sum(-1).mean()
                 for w in self.matryoshka_widths[:-1]]
        if inner:
            total = (full + self.inner_weight * sum(inner) / len(inner)) / (1.0 + self.inner_weight)
        else:
            total = full
        return total, {"full_recon": full.item()}


def train_sae(sae, X, steps=12000, bs=4096, lr=1e-3, seed=0, log=""):
    """clean_sae_pipeline loop: init b_dec=mean act, Adam, normalize_decoder each
    step, report FVU + dead-frac."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    with torch.no_grad():
        sae.b_dec.copy_(X.mean(0))
    with torch.enable_grad():
        opt = torch.optim.Adam(sae.parameters(), lr=lr)
        for step in range(steps):
            idx = rng.integers(0, N, bs)
            x = X[idx]
            loss, _ = sae.compute_loss(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                sae.normalize_decoder()
            if step % 4000 == 0:
                print(f"   [{log}] step {step} loss={loss.item():.4f}", flush=True)
    sae.eval()
    # FVU + dead-frac on the full set (chunked)
    with torch.no_grad():
        num = den = 0.0
        fired = torch.zeros(sae.d_sae, dtype=torch.bool)
        for s0 in range(0, N, 8192):
            x = X[s0:s0 + 8192]
            xh, z = sae.forward(x)
            num += (x - xh).pow(2).sum().item()
            den += (x - x.mean(0)).pow(2).sum().item()
            fired |= (z != 0).any(0)
        fvu = num / max(den, 1e-9)
        dead = float((~fired).float().mean().item())
    return fvu, dead


# ═══════════════════════ 2. planted-hierarchy generator ═════════════════
# Extends synth_hier2.py: ONE concept C with m drifting leaves selected by
# context; a single planted C->D attention route in the WEIGHTS; per-token
# ground-truth labels; superpos knob.
class HierGen:
    def __init__(self, alpha=0.6, superpos=1, seed=0, d=64, n=256, T=32,
                 m_ctx=8, p_mix=0.15, c_anchor=0.5):
        self.alpha, self.superpos, self.seed = alpha, superpos, seed
        self.d, self.n, self.T = d, n, T
        self.m_ctx, self.p_mix = m_ctx, p_mix
        # c_anchor: weak SHARED coarse-C signal co-injected with the (unit, dominant)
        # drifting leaf at the final query. It lets the Matryoshka prefix learn a
        # clean coarse-C feature (recovery gate) WITHOUT killing the flat SAE's
        # leaf-dominated drift (leaf weight 1.0 >> c_anchor). Design §1.3 / §5.2.1.
        self.c_anchor = c_anchor
        g = torch.Generator().manual_seed(seed)

        def randu(nn_):
            v = torch.randn(nn_, d, generator=g)
            return v / v.norm(dim=1, keepdim=True)

        self.K = 4
        self.idxC = 0
        self.idxC_leaf = list(range(1, 9))          # 8 leaves of C
        self.idxD = [9, 10, 11, 12]                  # 4 target domains
        self.idxDch = {k: list(range(13 + k * 16, 13 + (k + 1) * 16)) for k in range(self.K)}
        base = randu(n)
        F = base.clone()
        a = alpha
        # C leaves alpha-mixed toward C
        for c in self.idxC_leaf:
            F[c] = a * base[self.idxC] + np.sqrt(1 - a ** 2) * base[c]
        # D-domain leaves alpha-mixed toward their domain root
        for k in range(self.K):
            for c in self.idxDch[k]:
                F[c] = a * base[self.idxD[k]] + np.sqrt(1 - a ** 2) * base[c]
        F = F / F.norm(dim=1, keepdim=True)
        self.base, self.F = base, F
        rho = (F @ F.T).abs()
        self.rho = rho[~torch.eye(n, dtype=bool)].mean().item()
        lam = 0.1
        self.Ahat = F @ torch.linalg.inv(F.T @ F + lam * torch.eye(d))  # ground-truth read-outs
        # headline planted association: C <-> D[0]
        self.leaves_D = self.idxDch[0]
        # context g -> leaf c_{g+1}, p_mix chance of a different C-leaf (the drift)
        self.beta0 = 5.0

    def rd(self, x, i):
        return x @ self.Ahat[i]

    def persona_off_x(self, s):
        """Persona-off reference: remove the FULL concept contribution at the final
        query (the dominant drifting leaf + the weak shared coarse-C anchor). This
        is the removable (concept-driven) baseline R normalizes against."""
        x = s["x"].clone()
        x[self.T - 1] -= 1.0 * self.F[s["qleaf"]] + self.c_anchor * self.F[self.idxC]
        return x

    def gen(self, nseq, fp=None, fd=None, seed=0):
        """fp = force persona/concept-on; fd = force domain list. Records labels."""
        R = np.random.RandomState(seed)
        T, d, K = self.T, self.d, self.K
        F = self.F
        seqs = []
        for s in range(nseq):
            zper = fp if fp is not None else int(R.rand() < 0.5)
            ctx = int(R.randint(self.m_ctx))
            if fd is not None:
                doms = list(fd)
            else:
                doms = [int(R.randint(K))] if R.rand() < 0.5 else list(R.choice(K, 2, replace=False))
            x = torch.zeros(T, d)
            dompos = {k: [] for k in range(K)}
            cleafpos = []   # (t, leaf_index) for C-leaf emissions
            # the context's leaf realization of C (the DRIFT): leaf(ctx), or a
            # random other C-leaf w.p. p_mix (so the ctx->leaf map is not clean).
            qleaf = self.idxC_leaf[R.randint(self.m_ctx)] if R.rand() < self.p_mix else self.idxC_leaf[ctx]
            for t in range(1, T):
                r = R.rand()
                if r < 0.25 and doms:
                    k = doms[R.randint(len(doms))]
                    c = self.idxDch[k][R.randint(16)]
                    x[t] += R.uniform(0.8, 1.2) * F[c]
                    dompos[k].append(t)
                elif r < 0.40 and zper:
                    leaf = self.idxC_leaf[R.randint(self.m_ctx)] if R.rand() < self.p_mix else qleaf
                    x[t] += R.uniform(0.8, 1.2) * F[leaf]
                    cleafpos.append((t, leaf))
                else:
                    nbg = R.randint(1, 1 + self.superpos * 3)
                    for _ in range(nbg):
                        x[t] += R.uniform(0.8, 1.2) * F[int(R.randint(80, self.n))]
            # CRUCIAL (the drift source): the final-query CONCEPT realization is the
            # context's LEAF, not the fixed coarse anchor. Each leaf is alpha-mixed
            # toward C, so the head's read-out aC still fires (route lives), but the
            # flat SAE's dominant query feature DRIFTS across contexts; the
            # Matryoshka coarse prefix recovers the stable shared C direction.
            if zper:
                x[T - 1] += 1.0 * F[qleaf]                       # drifting leaf (dominant)
                x[T - 1] += self.c_anchor * F[self.idxC]         # weak shared coarse-C anchor
            for k in doms:
                x[T - 1] += 1.0 * F[self.idxD[k]]      # tonic domain anchor (key)
            x += torch.tensor(0.02 * R.randn(T, d), dtype=torch.float32)
            seqs.append(dict(x=x, zper=zper, doms=doms, dompos=dompos,
                             ctx=ctx, cleafpos=cleafpos, qleaf=qleaf))
        return seqs

    # ---- the planted single-linear-QK head (synth_hier2 build_WQK, K domains) ----
    # aC / bD are the NORMALIZED ground-truth read-outs; the head AND the oracle
    # cut both use them so the route subtraction is exactly self-consistent.
    def _route_dirs(self):
        aC = self.Ahat[self.idxC] / self.Ahat[self.idxC].norm()
        bD = [self.Ahat[self.idxD[k]] / self.Ahat[self.idxD[k]].norm() for k in range(self.K)]
        return aC, bD

    def build_head(self, sig):
        d, dh, K = self.d, 16, self.K
        E = torch.eye(dh)[:K]
        aC, bD = self._route_dirs()
        WQ = torch.zeros(d, dh)
        WK = torch.zeros(d, dh)
        for k in range(K):
            WQ += np.sqrt(sig) * torch.outer(aC, E[k])
            WK += np.sqrt(sig) * torch.outer(bD[k], E[k])
        return WQ, WK

    def raw_scores(self, x, WQ, WK):
        T = self.T
        S = (x @ WQ) @ (x @ WK).T
        S[:, 0] += self.beta0
        return S.masked_fill(torch.tril(torch.ones(T, T)) == 0, -1e9)

    @staticmethod
    def attn(S):
        return torch.softmax(S, -1)


def calibrate_sigma(G, kdom=0):
    """Pick sigma that MAXIMIZES the persona-on/off C->D[kdom] attention-mass GAP
    (the route's behavioral signal). The richer drift generator caps absolute
    domain-mass well below synth_hier2's 0.8, so we calibrate on the gap (the
    persona-driven removable part) — exactly what R normalizes against — and
    verify R_oracle ~ 1 downstream (the §5.2.3 guard)."""
    cal_on = G.gen(400, fp=1, fd=[kdom], seed=11)

    def dmass(seqs, WQ, WK):
        m = []
        for s in seqs:
            if kdom not in s["doms"] or not s["dompos"][kdom]:
                continue
            A = G.attn(G.raw_scores(s["x"], WQ, WK))
            m.append(sum(A[G.T - 1, t].item() for t in s["dompos"][kdom]))
        return float(np.mean(m)) if m else 0.0

    off_seqs = [dict(s, x=G.persona_off_x(s)) for s in cal_on]
    best = None
    for sig in [6, 10, 15, 20, 28, 40, 60]:
        WQ, WK = G.build_head(sig)
        on = dmass(cal_on, WQ, WK)
        off = dmass(off_seqs, WQ, WK)
        gap = on - off
        if best is None or gap > best[1]:
            best = (sig, gap, on)
    return best[0], best[2]


# ═══════════════════════ 3. FRA cell decomposition ══════════════════════
# Direct 4-D FRA for the synthetic head: S(q,k)=Σ_{μν} u^μ_q u^ν_k ω_{μν} with
# u^μ_q = z_q[μ]·(W_dec[μ]·WQ), u^ν_k = z_k[ν]·(W_dec[ν]·WK). We aggregate to a
# feature×feature cell matrix C[μ,ν] for the planted C->D edge: query = final
# position T-1, keys = D[0] domain positions. Averaged over sequences.
def fra_cell_matrix(G, sae, WQ, WK, seqs, kdom=0):
    """Returns C[d_sae, d_sae] : mean (over seqs, over D-domain key positions) of
    the FRA contribution to the final-query attention SCORE onto D-domain keys."""
    d_sae = sae.d_sae
    Wdec = sae.W_dec.detach().float()        # [d_sae, d]
    Q = Wdec @ WQ                            # [d_sae, dh]  per-feature query proj
    Kp = Wdec @ WK                           # [d_sae, dh]  per-feature key proj
    Cmat = torch.zeros(d_sae, d_sae)
    nseq = 0
    for s in seqs:
        if kdom not in s["doms"]:
            continue
        x = s["x"]
        z = sae.encode(x)                    # [T, d_sae]
        zq = z[G.T - 1]                       # final-query latent
        kpos = s["dompos"][kdom]
        if not kpos:
            continue
        uq = (zq.unsqueeze(-1) * Q)          # [d_sae, dh]
        acc = torch.zeros(d_sae, d_sae)
        for t in kpos:
            zk = z[t]
            uk = (zk.unsqueeze(-1) * Kp)     # [d_sae, dh]
            acc += uq @ uk.T                  # [d_sae, d_sae] contribution to score
        Cmat += acc / len(kpos)
        nseq += 1
    if nseq:
        Cmat /= nseq
    return Cmat


def recover_idx(direction, Wdec):
    """Best-matching decoder column by |cos|; returns (idx, |cos|)."""
    Wn = Wdec / Wdec.norm(dim=1, keepdim=True).clamp(min=1e-9)
    dn = direction / direction.norm().clamp(min=1e-9)
    cs = (Wn @ dn).abs()
    return int(cs.argmax()), float(cs.max())


# ═══════════════════════ 4. causal cut machinery ════════════════════════
# Cut SAE cells from the raw score: subtract the per-feature bilinear route
# Σ_{(μ,ν)∈cells} z_q[μ] z_k[ν] (W_dec[μ]·WQ)(W_dec[ν]·WK). Reproduces
# synth_hier2.cell_S but on SAE features instead of Ahat rows. Then re-measure
# the C->D attention mass (ground-truth behavioral readout), normalized to the
# persona-off baseline => removal R (= synth_hier2.measure's R_X).
def make_cut_Sfn(G, sae, WQ, WK, cells):
    """cells: set of (qfeat, kfeat) latent index pairs to subtract."""
    Wdec = sae.W_dec.detach().float()
    Q = Wdec @ WQ
    Kp = Wdec @ WK
    cellset = set(cells)

    def Sfn(x):
        z = sae.encode(x)                    # [T, d_sae]
        S = G.raw_scores(x, WQ, WK)
        # Direct route subtraction: for each cell, contribution(q,k)=z_q[a] z_k[b] (Q[a]·Kp[b])
        sub = torch.zeros(G.T, G.T)
        for (a, b) in cellset:
            w = float(Q[a] @ Kp[b])
            sub += (z[:, a].unsqueeze(1) * z[:, b].unsqueeze(0)) * w
        S2 = S - sub
        return S2.masked_fill(torch.tril(torch.ones(G.T, G.T)) == 0, -1e9)

    return Sfn


def cd_mass(G, x, doms_seq, kdom, WQ, WK, Sfn=None):
    """Final-query attention mass onto D-domain (kdom) key positions."""
    A = G.attn(Sfn(x) if Sfn else G.raw_scores(x, WQ, WK))
    return sum(A[G.T - 1, t].item() for t in doms_seq[kdom])


def removal(G, sae, WQ, WK, seqs, cells, kdom=0):
    """R = (base - cut)/(base - off) for the C->D mass, averaged over seqs.
    off = persona-off baseline (subtract tonic concept anchor)."""
    Sfn = make_cut_Sfn(G, sae, WQ, WK, cells) if cells else None
    base = []
    cut = []
    off = []
    for s in seqs:
        if kdom not in s["doms"] or not s["dompos"][kdom]:
            continue
        base.append(cd_mass(G, s["x"], s["dompos"], kdom, WQ, WK))
        cut.append(cd_mass(G, s["x"], s["dompos"], kdom, WQ, WK, Sfn=Sfn) if cells else base[-1])
        off.append(cd_mass(G, G.persona_off_x(s), s["dompos"], kdom, WQ, WK))
    base, cut, off = np.mean(base), np.mean(cut), np.mean(off)
    return float((base - cut) / max(base - off, 1e-6))


def collateral(G, sae, WQ, WK, seqs, cells, kdom=0, kother=1):
    """Collateral: removal applied to a DIFFERENT active domain (specificity)."""
    Sfn = make_cut_Sfn(G, sae, WQ, WK, cells) if cells else None
    base = []
    cut = []
    off = []
    for s in seqs:
        if kother not in s["doms"] or not s["dompos"][kother]:
            continue
        base.append(cd_mass(G, s["x"], s["dompos"], kother, WQ, WK))
        cut.append(cd_mass(G, s["x"], s["dompos"], kother, WQ, WK, Sfn=Sfn) if cells else base[-1])
        off.append(cd_mass(G, G.persona_off_x(s), s["dompos"], kother, WQ, WK))
    if not base:
        return 0.0
    base, cut, off = np.mean(base), np.mean(cut), np.mean(off)
    return float(abs(base - cut) / max(abs(base - off), 1e-6))


# ═══════════════════════ per-(alpha,superpos,seed) cell ═════════════════
# DRIFT-VALIDITY FIX (orchestrator, 2026-06-12): the v1 SMOKE run (alphas 0.5-0.9,
# superpos 1, seed 0, C_ANCHOR 0.5) GATE-FAILED in every cell on flat_drift:
# top1_coverage_flat=1.0 (NO drift) vs the gpt2 anchor 0.32. Root cause: the flat
# SAE latches the CONCEPT, not the drifting leaf, whenever concept weight at the
# query (alpha + C_ANCHOR) exceeds the drifting-leaf weight sqrt(1-alpha^2). With
# C_ANCHOR=0.5 only alpha<~0.35 drifts; the smoke alphas (>=0.5) could NOT drift.
# Lowering C_ANCHOR to 0.2 extends the drift regime to alpha<~0.57, so the default
# alpha sweep 0.4..0.7 now BRACKETS the drift onset (0.4/0.5 drift, 0.6/0.7 do not)
# -> the flat baseline reproduces the gpt2 drift pathology (the VALIDITY precondition)
# BEFORE the hierarchical-SAE remedy is tested. This restores the design intent
# (drifting leaf dominant; c_anchor a WEAK shared signal for the Matryoshka prefix);
# the CONFIRM/FALSIFY verdict still reads off the coarse-cut residual, NOT C_ANCHOR.
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.4,0.5,0.6,0.7").split(",")]
SUPERPOS = [int(s) for s in os.environ.get("SUPERPOS", "1,2").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
STEPS = int(os.environ.get("STEPS", "10000"))
N_TRAIN = int(os.environ.get("N_TRAIN", "4000"))
M_COARSE = int(os.environ.get("M_COARSE", "16"))
M_SAE = int(os.environ.get("M_SAE", "256"))
K_TOPK = int(os.environ.get("K_TOPK", "8"))
INNER_WEIGHT = float(os.environ.get("INNER_WEIGHT", "2.0"))
C_ANCHOR = float(os.environ.get("C_ANCHOR", "0.2"))

res = {"config": dict(alphas=ALPHAS, superpos=SUPERPOS, seeds=SEEDS, steps=STEPS,
                      n_train=N_TRAIN, m_coarse=M_COARSE, m_sae=M_SAE, k_topk=K_TOPK,
                      inner_weight=INNER_WEIGHT, c_anchor=C_ANCHOR),
       "cells": [], "done": False}


def ckpt(done=False):
    res["done"] = done
    OUT.write_text(json.dumps(res, indent=2, default=float))


def run_cell(alpha, superpos, seed):
    G = HierGen(alpha=alpha, superpos=superpos, seed=seed, c_anchor=C_ANCHOR)
    sig, on_mass = calibrate_sigma(G)
    WQ, WK = G.build_head(sig)

    # ---- training data ----
    train_seqs = G.gen(N_TRAIN, fp=None, seed=1000 + seed)
    X = torch.cat([s["x"] for s in train_seqs], 0)        # [N_TRAIN*T, d]

    # ---- two SAEs ----
    flat = TopKSAE(d_in=G.d, d_sae=M_SAE, k=K_TOPK)
    matry = MatryoshkaSAE(d_in=G.d, d_sae=M_SAE, k=K_TOPK,
                          matryoshka_widths=[M_COARSE, M_SAE], inner_weight=INNER_WEIGHT)
    fvu_flat, dead_flat = train_sae(flat, X, steps=STEPS, seed=seed, log=f"flat a{alpha}s{superpos}r{seed}")
    fvu_matry, dead_matry = train_sae(matry, X, steps=STEPS, seed=seed, log=f"matry a{alpha}s{superpos}r{seed}")

    # ---- recovery check (§2.3) ----
    Wf = flat.W_dec.detach().float()
    Wm = matry.W_dec.detach().float()
    # flat: leaves recovered? coarse NOT dedicated?
    leafcos_flat = [recover_idx(G.F[c], Wf)[1] for c in G.idxC_leaf]
    coarseC_flat = recover_idx(G.base[G.idxC], Wf)
    # matry: coarse concept at index<M_COARSE?
    coarseC_m = recover_idx(G.base[G.idxC], Wm[:M_COARSE])
    coarseD_m = recover_idx(G.base[G.idxD[0]], Wm[:M_COARSE])
    coarse_C_idx = coarseC_m[0]                 # within prefix
    coarse_D_idx = coarseD_m[0]
    # fine leaves on matry tail
    fine_C_idxs = [recover_idx(G.F[c], Wm)[0] for c in G.idxC_leaf]
    fine_D_idxs = [recover_idx(G.F[c], Wm)[0] for c in G.leaves_D]
    recovery_matry_pass = bool(coarseC_m[1] >= 0.8)

    # ---- eval data: persona-on, headline domain D0 forced present (so the C->D0
    # mass is not split across domains); a mixed [0,1] set is used for collateral. ----
    ev = G.gen(800, fp=1, fd=[0], seed=2000 + seed)
    ev_mix = G.gen(600, fp=1, fd=[0, 1], seed=3000 + seed)

    # ---- planted union cells (ground truth) on each SAE basis ----
    def union_cells(Wdec):
        # map each planted leaf/concept direction to its best SAE latent
        qC = recover_idx(G.base[G.idxC], Wdec)[0]
        kD = recover_idx(G.base[G.idxD[0]], Wdec)[0]
        qleaves = [recover_idx(G.F[c], Wdec)[0] for c in G.idxC_leaf]
        kleaves = [recover_idx(G.F[c], Wdec)[0] for c in G.leaves_D]
        cells = set()
        cells.add((qC, kD))
        for a in set(qleaves + [qC]):
            for b in set(kleaves + [kD]):
                cells.add((a, b))
        return cells, qC, kD, set(qleaves), set(kleaves)

    fcells, qCf, kDf, qLf, kLf = union_cells(Wf)
    mcells, qCm, kDm, qLm, kLm = union_cells(Wm)

    # ---- FRA cell matrices ----
    Cf = fra_cell_matrix(G, flat, WQ, WK, ev, kdom=0)
    Cm = fra_cell_matrix(G, matry, WQ, WK, ev, kdom=0)

    # ════ §3.1 level-spreading (Matryoshka) ════
    coarse_mask = torch.zeros(M_SAE, dtype=torch.bool); coarse_mask[:M_COARSE] = True
    fine_mask = ~coarse_mask
    # restrict to the planted-union cells only (the association's score), abs
    Mabs = Cm.abs()
    union_idx = list(mcells)
    def block_sum(qm, km):
        s = 0.0
        for (a, b) in union_idx:
            if qm[a] and km[b]:
                s += float(Mabs[a, b])
        return s
    cc = block_sum(coarse_mask, coarse_mask)
    cf = block_sum(coarse_mask, fine_mask)
    fc = block_sum(fine_mask, coarse_mask)
    ff = block_sum(fine_mask, fine_mask)
    tot = cc + cf + fc + ff
    frac_coarse_xx = float(cc / max(tot, 1e-12))
    frac_coarse_fine = float(cf / max(tot, 1e-12))
    frac_fine_coarse = float(fc / max(tot, 1e-12))
    frac_fine_fine = float(ff / max(tot, 1e-12))

    # ════ §3.2 coarse-cut residual (Matryoshka, causal) ════
    R_coarse_only = removal(G, matry, WQ, WK, ev, {(qCm, kDm)}, kdom=0)
    # coarse + cross-level (coarse_C with fine D-keys, fine C-queries with coarse D)
    cross_cells = {(qCm, kDm)}
    for b in kLm:
        cross_cells.add((qCm, b))
    for a in qLm:
        cross_cells.add((a, kDm))
    R_coarse_plus_cross = removal(G, matry, WQ, WK, ev, cross_cells, kdom=0)
    R_full_union_m = removal(G, matry, WQ, WK, ev, mcells, kdom=0)
    # oracle: cut the ground-truth Ahat route (full removal ceiling)
    R_oracle = oracle_removal(G, WQ, WK, ev, sig, kdom=0)
    residual_coarse = float(R_full_union_m - R_coarse_only)
    carry_coarse = float(R_coarse_only / max(R_full_union_m, 1e-6))

    # ---- flat comparator: top1 cell + n_cells_for_90 ----
    R_full_union_f = removal(G, flat, WQ, WK, ev, fcells, kdom=0)
    flat_top1, flat_drift, n90_flat = flat_localization(G, flat, WQ, WK, ev, fcells, R_full_union_f)
    matry_top1cov = matry_localization(G, matry, WQ, WK, ev, qCm, kDm)

    # ════ §4.1 DETECTION (AUROC of FRA cell-block magnitude vs planted graph) ════
    det_flat = detection_auroc(G, Cf, Wf)
    det_matry = detection_auroc(G, Cm, Wm)

    # ════ §4.3 CONTROL: held-out collateral + specificity (on the MIXED [0,1] set) ════
    collat_coarse = collateral(G, matry, WQ, WK, ev_mix, {(qCm, kDm)}, kdom=0, kother=1)
    collat_union_m = collateral(G, matry, WQ, WK, ev_mix, mcells, kdom=0, kother=1)
    # embedding baseline: project out concept C everywhere
    R_emb, collat_emb = embedding_baseline(G, WQ, WK, ev_mix, kdom=0, kother=1)
    spec_ratio = float(collat_emb / max(collat_coarse, 1e-6))

    return dict(
        alpha=alpha, superpos=superpos, seed=seed, sig=sig, on_mass=on_mass, rho=G.rho,
        fvu_flat=fvu_flat, fvu_matry=fvu_matry, dead_flat=dead_flat, dead_matry=dead_matry,
        recovery=dict(
            leafcos_flat_min=float(min(leafcos_flat)), leafcos_flat_mean=float(np.mean(leafcos_flat)),
            coarseC_flat_cos=coarseC_flat[1],
            coarseC_matry_cos=coarseC_m[1], coarseD_matry_cos=coarseD_m[1],
            coarse_C_idx=coarse_C_idx, coarse_D_idx=coarse_D_idx,
            recovery_matry_pass=recovery_matry_pass,
            flat_drift_pass=bool(flat_top1 < 0.5),
        ),
        level_spreading=dict(frac_coarse_xx=frac_coarse_xx, frac_coarse_fine=frac_coarse_fine,
                             frac_fine_coarse=frac_fine_coarse, frac_fine_fine=frac_fine_fine),
        coarse_cut=dict(R_coarse_only=R_coarse_only, R_coarse_plus_cross=R_coarse_plus_cross,
                        R_full_union_m=R_full_union_m, R_full_union_f=R_full_union_f,
                        R_oracle=R_oracle, residual_coarse=residual_coarse,
                        carry_coarse=carry_coarse),
        detection=dict(det_auroc_flat=det_flat, det_auroc_matry=det_matry),
        localization=dict(top1_coverage_flat=flat_top1, top1_coverage_matry=matry_top1cov,
                          n_cells_for_90_flat=n90_flat),
        control=dict(collat_coarse=collat_coarse, collat_union_matry=collat_union_m,
                     R_emb=R_emb, collat_emb=collat_emb, spec_ratio=spec_ratio),
    )


# ──────────────────── helper metric functions ───────────────────────────
def oracle_removal(G, WQ, WK, seqs, sig, kdom=0):
    """Cut the ground-truth C->D route EXACTLY (self-consistent with build_head:
    route(q,k) = sig*(x_q·aC)(x_k·bD[kdom]) with the SAME normalized read-outs)."""
    aC, bD = G._route_dirs()
    def Sfn(x):
        S = G.raw_scores(x, WQ, WK)
        sub = sig * torch.outer(x @ aC, x @ bD[kdom])
        return (S - sub).masked_fill(torch.tril(torch.ones(G.T, G.T)) == 0, -1e9)
    base, cut, off = [], [], []
    for s in seqs:
        if kdom not in s["doms"] or not s["dompos"][kdom]:
            continue
        base.append(cd_mass(G, s["x"], s["dompos"], kdom, WQ, WK))
        cut.append(cd_mass(G, s["x"], s["dompos"], kdom, WQ, WK, Sfn=Sfn))
        off.append(cd_mass(G, G.persona_off_x(s), s["dompos"], kdom, WQ, WK))
    base, cut, off = np.mean(base), np.mean(cut), np.mean(off)
    return float((base - cut) / max(base - off, 1e-6))


def embedding_baseline(G, WQ, WK, seqs, kdom=0, kother=1):
    """Project out the COARSE concept C direction everywhere (position-invariant,
    association-BLIND baseline — kills all of C, not just C->D)."""
    fC = G.F[G.idxC]
    def proj(x):
        return x - (x @ fC).unsqueeze(-1) * fC
    def rem(kk):
        base, cut, off = [], [], []
        for s in seqs:
            if kk not in s["doms"] or not s["dompos"][kk]:
                continue
            base.append(cd_mass(G, s["x"], s["dompos"], kk, WQ, WK))
            cut.append(cd_mass(G, proj(s["x"]), s["dompos"], kk, WQ, WK))
            off.append(cd_mass(G, G.persona_off_x(s), s["dompos"], kk, WQ, WK))
        base, cut, off = np.mean(base), np.mean(cut), np.mean(off)
        return float((base - cut) / max(base - off, 1e-6)), base, cut, off
    R, b, c, o = rem(kdom)
    # collateral on other domain
    cb, cc_, co = [], [], []
    for s in seqs:
        if kother not in s["doms"] or not s["dompos"][kother]:
            continue
        cb.append(cd_mass(G, s["x"], s["dompos"], kother, WQ, WK))
        cc_.append(cd_mass(G, proj(s["x"]), s["dompos"], kother, WQ, WK))
        co.append(cd_mass(G, G.persona_off_x(s), s["dompos"], kother, WQ, WK))
    if cb:
        collat = float(abs(np.mean(cb) - np.mean(cc_)) / max(abs(np.mean(cb) - np.mean(co)), 1e-6))
    else:
        collat = 0.0
    return R, collat


def flat_localization(G, sae, WQ, WK, seqs, fcells, R_full):
    """top1_coverage across contexts + n_cells_for_90 for the flat SAE."""
    # per-context dominant FRA cell on the C->D edge
    by_ctx = {}
    for s in seqs:
        by_ctx.setdefault(s["ctx"], []).append(s)
    top1_per_ctx = {}
    for ctx, ss in by_ctx.items():
        C = fra_cell_matrix(G, sae, WQ, WK, ss, kdom=0).abs()
        # restrict to union cells
        best = None; bestv = -1
        for (a, b) in fcells:
            if float(C[a, b]) > bestv:
                bestv = float(C[a, b]); best = (a, b)
        top1_per_ctx[ctx] = best
    # coverage = fraction of contexts sharing the modal top1 cell
    from collections import Counter
    cnt = Counter(top1_per_ctx.values())
    modal, modal_n = cnt.most_common(1)[0]
    top1_coverage = modal_n / max(len(top1_per_ctx), 1)
    # n_cells_for_90: rank union cells by pooled |FRA|, count to reach 90% of R_full via causal cut
    Cpool = fra_cell_matrix(G, sae, WQ, WK, seqs, kdom=0).abs()
    ranked = sorted(fcells, key=lambda ab: -float(Cpool[ab[0], ab[1]]))
    n90 = len(ranked)
    cum = set()
    for i, ab in enumerate(ranked, 1):
        cum.add(ab)
        r = removal(G, sae, WQ, WK, seqs, cum, kdom=0)
        if r >= 0.90 * R_full:
            n90 = i; break
    return float(top1_coverage), modal, int(n90)


def matry_localization(G, sae, WQ, WK, seqs, qC, kD):
    """top1_coverage for the Matryoshka: how often the coarse (qC,kD) cell is the
    modal top-1 FRA cell across contexts (within the union)."""
    Wdec = sae.W_dec.detach().float()
    _, fc = recover_idx(G.base[G.idxC], Wdec)
    # union cells for matry
    qleaves = [recover_idx(G.F[c], Wdec)[0] for c in G.idxC_leaf]
    kleaves = [recover_idx(G.F[c], Wdec)[0] for c in G.leaves_D]
    cells = set([(qC, kD)])
    for a in set(qleaves + [qC]):
        for b in set(kleaves + [kD]):
            cells.add((a, b))
    by_ctx = {}
    for s in seqs:
        by_ctx.setdefault(s["ctx"], []).append(s)
    hit = 0
    for ctx, ss in by_ctx.items():
        C = fra_cell_matrix(G, sae, WQ, WK, ss, kdom=0).abs()
        best = None; bestv = -1
        for (a, b) in cells:
            if float(C[a, b]) > bestv:
                bestv = float(C[a, b]); best = (a, b)
        if best == (qC, kD):
            hit += 1
    return float(hit / max(len(by_ctx), 1))


def detection_auroc(G, Cmat, Wdec):
    """AUROC of FRA cell-block magnitude vs planted concept-graph.
    Positive edge: C->D[0]. Negatives: C->D[1:], C->background-domain.
    Score for pair (X-features, Y-features) = sum |Cmat| over those feature sets."""
    Mabs = Cmat.abs()
    qC = recover_idx(G.base[G.idxC], Wdec)[0]

    def block(qfeat, kfeats):
        return float(sum(float(Mabs[qfeat, b]) for b in kfeats))

    pos, neg = [], []
    # query = C; keys = each domain's recovered feature set
    for k in range(G.K):
        kfeats = list({recover_idx(G.base[G.idxD[k]], Wdec)[0]} |
                      {recover_idx(G.F[c], Wdec)[0] for c in G.idxDch[k]})
        sc = block(qC, kfeats)
        if k == 0:
            pos.append(sc)
        else:
            neg.append(sc)
    # background negatives: a few random background features as keys
    rng = np.random.default_rng(0)
    bg = rng.integers(80, G.n, 8)
    bgfeats = list({recover_idx(G.F[int(c)], Wdec)[0] for c in bg})
    neg.append(block(qC, bgfeats))
    # AUROC = fraction of (pos,neg) pairs with pos>neg
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for nn_ in neg if p > nn_) + 0.5 * sum(1 for p in pos for nn_ in neg if p == nn_)
    return float(wins / (len(pos) * len(neg)))


# ═══════════════════════════ run the sweep ══════════════════════════════
def verdict_for_cell(c):
    """Apply locked §5.1 thresholds (CONFIRM/FALSIFY/HELP) + the §5.2/5.3 guards."""
    rec = c["recovery"]; ls = c["level_spreading"]; cc = c["coarse_cut"]
    gate_matry = rec["recovery_matry_pass"]
    gate_drift = rec["flat_drift_pass"]
    gate_union = cc["R_full_union_m"] >= 0.7 * max(cc["R_oracle"], 1e-6)  # full union ~ oracle
    fcx = ls["frac_coarse_xx"]; resid = cc["residual_coarse"]; carry = cc["carry_coarse"]
    top1m = c["localization"]["top1_coverage_matry"]
    R_co = cc["R_coarse_only"]; R_fu = cc["R_full_union_m"]
    collat_ok_falsify = c["control"]["collat_coarse"] <= c["control"]["collat_emb"]
    gates = dict(recovery_matry=gate_matry, flat_drift=gate_drift, full_union_ge_oracle=gate_union)
    if not (gate_matry and gate_drift):
        return "GATE-FAIL", gates
    # FALSIFY: clean single-cell win
    if (R_co >= 0.70 * max(R_fu, 1e-6)) and (fcx >= 0.70) and (top1m >= 0.80) and collat_ok_falsify:
        return "FALSIFY", gates
    # CONFIRM: split + residual + bounded union
    if (resid >= 0.30 or carry < 0.60) and (fcx < 0.60) and (R_co < R_fu):
        return "CONFIRM", gates
    return "HELP-BUT-NOT-SOLVE", gates


t0 = time.time()
ckpt()
print(f"[hiersae] sweep: alphas={ALPHAS} superpos={SUPERPOS} seeds={SEEDS} "
      f"steps={STEPS} m_coarse={M_COARSE} m_sae={M_SAE} k={K_TOPK}", flush=True)
for alpha in ALPHAS:
    for sp in SUPERPOS:
        for seed in SEEDS:
            try:
                c = run_cell(alpha, sp, seed)
                v, gates = verdict_for_cell(c)
                c["verdict"] = v
                c["gates"] = gates
                res["cells"].append(c)
                ckpt()   # durable LOCAL progress; HF upload only at END (commit-rate cap)
                print(f"[cell] a={alpha} sp={sp} seed={seed} -> {v} | "
                      f"fcx={c['level_spreading']['frac_coarse_xx']:.2f} "
                      f"resid={c['coarse_cut']['residual_coarse']:.2f} "
                      f"R_co={c['coarse_cut']['R_coarse_only']:.2f} "
                      f"R_full={c['coarse_cut']['R_full_union_m']:.2f} "
                      f"R_orc={c['coarse_cut']['R_oracle']:.2f} "
                      f"top1f={c['localization']['top1_coverage_flat']:.2f} "
                      f"top1m={c['localization']['top1_coverage_matry']:.2f} "
                      f"gates={gates} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[cell ERROR] a={alpha} sp={sp} seed={seed}: {e}\n{tb}", flush=True)
                res["cells"].append(dict(alpha=alpha, superpos=sp, seed=seed, error=str(e), tb=tb))
                ckpt()

ckpt(True)
_upload(OUT, f"{PFX}/{OUT.name}")   # name-derived: drift re-run won't clobber the smoke result on HF
print(f"[hiersae] DONE {len(res['cells'])} cells in {time.time()-t0:.0f}s", flush=True)
