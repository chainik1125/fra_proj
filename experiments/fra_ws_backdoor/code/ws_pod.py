#!/usr/bin/env python3
"""B1 weight-sparse x FRA pod script (stages A/B/C, resumable, partial-upload).

Stage A: load the 3-model ladder, verify quote/binding tasks, find the QK head,
         NEURON-BASIS FRA (act_in + bias channel; exact, SAE-free) -> concentration/
         drift/set-drift metrics + causal Spearman/recovery vs oracle.
Stage B: identical TopK SAEs on act_in at the circuit layer, same Python corpus
         -> FVU / dead-frac / L0 / R2 / monosemanticity proxies.
Stage C: SAE-BASIS FRA on the same prompts/edges -> same metrics + SAE-QK recon R2.

Models (OpenAI circuit-sparsity, Gao et al. 2025, arXiv 2511.13653):
  sparse = csp_sweep1_1x_3.7Mnonzero_afrac0.250   (weight-sparse + act-sparse)
  wsda   = csp_sweep1_1x_3.7Mnonzero_afrac1.000   (same weights-budget, dense acts)
  dense  = dense1_1x (fully dense; depth-4 confound flagged; fallback dense1_2x)

HF artifacts: dmanningcoe/fra-phase1-steering-data : fra_weightsparse/results/
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import random
import sys
import time
import traceback
import types

import numpy as np
import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csp_vendor_gpt as G
from csp_vendor_hook_utils import hook_recorder
from sae_models_ws import TopKSAE

# ---------------- config ----------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", "0"))
STAGES = os.environ.get("STAGES", "ABC")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/out")
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
HF_PREFIX = "fra_weightsparse/results"
BLOB = "https://openaipublic.blob.core.windows.net/circuit-sparsity/models"

MODELS = {
    "sparse": "csp_sweep1_1x_3.7Mnonzero_afrac0.250",
    "wsda": "csp_sweep1_1x_3.7Mnonzero_afrac1.000",
    "dense": "dense1_1x",
}
DENSE_FALLBACK = "dense1_2x"
TASK_GATE = float(os.environ.get("TASK_GATE", "0.70"))  # min accuracy to accept a model on a task

N_QUOTE = int(os.environ.get("N_QUOTE", "200"))      # 100 locate / 100 holdout
N_BIND = int(os.environ.get("N_BIND", "120"))        # 60 locate / 60 holdout
N_CAUSAL_TOP = int(os.environ.get("N_CAUSAL_TOP", "150"))
N_CAUSAL_RAND = int(os.environ.get("N_CAUSAL_RAND", "150"))

SAE_D = int(os.environ.get("SAE_D", "2048"))
SAE_K = int(os.environ.get("SAE_K", "32"))
SAE_STEPS = int(os.environ.get("SAE_STEPS", "12000"))
SAE_BS = int(os.environ.get("SAE_BS", "4096"))
SAE_LR = float(os.environ.get("SAE_LR", "3e-4"))
N_TRAIN_TOK = int(os.environ.get("N_TRAIN_TOK", "2000000"))
N_EVAL_TOK = int(os.environ.get("N_EVAL_TOK", "200000"))

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------- HF up/down ----------------
def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def upload(path, name=None):
    name = name or os.path.basename(path)
    for i in range(5):
        try:
            hf_api().upload_file(path_or_fileobj=path, path_in_repo=f"{HF_PREFIX}/{name}",
                                 repo_id=HF_REPO, repo_type="dataset")
            return True
        except Exception as e:
            print(f"[upload] {name} attempt {i}: {e}", flush=True)
            time.sleep(30 * (i + 1))
    return False


def try_download(name, dest):
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(HF_REPO, f"{HF_PREFIX}/{name}", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
        import shutil
        shutil.copy(p, dest)
        return True
    except Exception:
        return False


def save_stage(stage, obj):
    p = os.path.join(OUT_DIR, f"ws_stage{stage}.json")
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    upload(p)


# ---------------- model loading ----------------
def fetch_url(url, binary=False):
    for i in range(6):
        try:
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception as e:
            print(f"[fetch] {url} attempt {i}: {e}", flush=True)
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}")


def load_ws_model(name):
    cfg = json.loads(fetch_url(f"{BLOB}/{name}/beeg_config.json"))
    if "n_mlp" in cfg:
        cfg["d_mlp"] = cfg.pop("n_mlp")
    fields = {f.name for f in dataclasses.fields(G.GPTConfig)}
    cfg = {k: v for k, v in cfg.items() if k in fields}
    cfg["flash"] = False           # manual attention path (needed for logit edits)
    cfg["grad_checkpointing"] = False
    cfg.setdefault("sink", False)
    assert not cfg["sink"], f"{name} has a sink; unsupported with flash=False"
    config = G.GPTConfig(**cfg)
    model = G.GPT(config)
    ckpt_cache = os.path.join(OUT_DIR, f"{name}.pt")
    if not os.path.exists(ckpt_cache):
        with open(ckpt_cache, "wb") as f:
            f.write(fetch_url(f"{BLOB}/{name}/final_model.pt", binary=True))
    sd = torch.load(ckpt_cache, weights_only=True, map_location="cpu")
    if "final_logits_bias" not in sd:
        sd["final_logits_bias"] = torch.zeros(config.vocab_size)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # tolerated-missing: the causal-mask buffer "transformer.h.<L>.attn.bias" (registered
    # only when flash=False; absent from checkpoints saved with flash=True).
    bad_missing = [m for m in missing
                   if not (m.startswith("transformer.h.") and m.endswith(".attn.bias"))]
    assert not unexpected, f"{name} unexpected keys: {unexpected[:8]}"
    assert not bad_missing, f"{name} missing keys: {bad_missing[:8]}"
    model.eval().to(DEV)
    print(f"[load] {name}: n_layer={config.n_layer} n_head={config.n_head} d_model={config.d_model} "
          f"afrac={config.afrac} loctypes={config.afrac_loctypes} missing={len(missing)}", flush=True)
    return model, config


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    p = hf_hub_download("openai/circuit-sparsity", "tokenizer.json")
    return Tokenizer.from_file(p)


# ---------------- tasks ----------------
IDENTS = ["x", "y", "val", "name", "out", "data", "msg", "path", "key", "item", "res", "tmp",
          "arg", "row", "txt", "buf", "tag", "src", "dst", "obj"]
WORDS = ["hello", "world", "alpha", "beta", "gamma", "delta", "result", "input", "output",
         "node", "graph", "list", "tree", "value", "error", "token", "index", "count"]
CARRIERS = [
    "def f():\n    {pre}{q}{content}",
    "import os\n\n{pre}{q}{content}",
    "for i in range(3):\n    pass\n{pre}{q}{content}",
    "class A:\n    pass\n\n{pre}{q}{content}",
    "{pre}{q}{content}",
    "if True:\n    {pre}{q}{content}",
    "# compute the thing\n{pre}{q}{content}",
    "z = [1, 2, 3]\n{pre}{q}{content}",
]
PRES = ["{v} = ", "print(", "raise ValueError(", "{v} = {{'k': ", "f({v}, ", "return "]


def gen_quote_prompts(n, rng):
    prompts = []
    for i in range(n):
        q = rng.choice(["'", '"'])
        carrier = rng.choice(CARRIERS)
        pre = rng.choice(PRES).format(v=rng.choice(IDENTS))
        nw = rng.randint(1, 4)
        content = " ".join(rng.choice(WORDS) for _ in range(nw))
        text = carrier.format(pre=pre, q=q, content=content)
        prompts.append({"text": text, "quote": q})
    return prompts


def gen_bind_prompts(n, rng):
    """set_or_string: v1 = set(); v2 = "" ; ... ; vq -> '.add(' vs ' += '."""
    prompts = []
    for i in range(n):
        v1, v2 = rng.sample(IDENTS, 2)
        set_first = rng.random() < 0.5
        lines = ([f"{v1} = set()", f'{v2} = ""'] if set_first
                 else [f'{v2} = ""', f"{v1} = set()"])
        nd = rng.randint(0, 2)
        for _ in range(nd):
            lines.append(f"{rng.choice([w for w in IDENTS if w not in (v1, v2)])} = {rng.randint(0, 99)}")
        query_set = rng.random() < 0.5
        vq = v1 if query_set else v2
        text = "\n".join(lines) + f"\n{vq}"
        prompts.append({"text": text, "query_set": query_set, "set_var": v1, "str_var": v2})
    return prompts


def pad_batch(tok_lists, pad_id=0):
    T = max(len(t) for t in tok_lists)
    ids = torch.full((len(tok_lists), T), pad_id, dtype=torch.long)
    for i, t in enumerate(tok_lists):
        ids[i, : len(t)] = torch.tensor(t)
    lens = torch.tensor([len(t) for t in tok_lists])
    return ids.to(DEV), lens.to(DEV)


_QSETS = {}


def quote_token_sets(tok):
    """ids whose DECODED string starts with " / ' (robust to byte-level vocab reprs)."""
    if "sets" not in _QSETS:
        dq, sq = [], []
        for i in range(tok.get_vocab_size()):
            d = tok.decode([i])
            if d.startswith('"'):
                dq.append(i)
            elif d.startswith("'"):
                sq.append(i)
        assert dq and sq, "no quote tokens found"
        _QSETS["sets"] = (torch.tensor(dq), torch.tensor(sq))
    return _QSETS["sets"]


@torch.no_grad()
def quote_scores(model, tok, prompts, batch=64):
    """score in [0,1]: P(correct quote-type continuation) / P(either)."""
    dq, sq = quote_token_sets(tok)
    scores = []
    for s0 in range(0, len(prompts), batch):
        chunk = prompts[s0 : s0 + batch]
        toks = [tok.encode(p["text"]).ids for p in chunk]
        ids, lens = pad_batch(toks)
        logits, _, _ = model(ids)
        for b, p in enumerate(chunk):
            lp = F.softmax(logits[b, lens[b] - 1].float(), dim=-1)
            p_dq, p_sq = lp[dq.to(DEV)].sum().item(), lp[sq.to(DEV)].sum().item()
            pc, pw = (p_dq, p_sq) if p["quote"] == '"' else (p_sq, p_dq)
            scores.append(pc / max(pc + pw, 1e-12))
    return np.array(scores)


@torch.no_grad()
def bind_scores(model, tok, prompts, batch=32):
    """score in [0,1] via teacher-forced logprob of '.add(' vs ' += '."""
    scores = []
    comps = {True: ".add(", False: " += "}
    for s0 in range(0, len(prompts), batch):
        chunk = prompts[s0 : s0 + batch]
        rows, metas = [], []
        for p in chunk:
            base = tok.encode(p["text"]).ids
            for is_set in (True, False):
                comp = tok.encode(comps[is_set]).ids
                rows.append(base + comp)
                metas.append((len(base), len(comp), is_set))
        ids, lens = pad_batch(rows)
        logits, _, _ = model(ids)
        logp = F.log_softmax(logits.float(), dim=-1)
        lps = {}
        for r, (nb, nc, is_set) in enumerate(metas):
            lp = sum(logp[r, nb - 1 + j, ids[r, nb + j]].item() for j in range(nc))
            lps.setdefault(r // 2, {})[is_set] = lp
        for j, p in enumerate(chunk):
            lp_set, lp_str = lps[j][True], lps[j][False]
            lc, lw = (lp_set, lp_str) if p["query_set"] else (lp_str, lp_set)
            scores.append(math.exp(lc) / max(math.exp(lc) + math.exp(lw), 1e-300))
    return np.array(scores)


def find_kpos_quote(tok, prompt):
    """position of the LAST token containing the opening quote char."""
    ids = tok.encode(prompt["text"]).ids
    pos = [i for i, t in enumerate(ids) if prompt["quote"] in tok.decode([t])]
    return ids, pos[-1]


def find_kpos_bind(tok, prompt):
    """position of the init-value token of the QUERIED var (set( ... or "" )."""
    ids = tok.encode(prompt["text"]).ids
    needle = "set(" if prompt["query_set"] else '""'
    text = prompt["text"]
    char_at = text.index(f"{(prompt['set_var'] if prompt['query_set'] else prompt['str_var'])} = ") \
        + len(f"{(prompt['set_var'] if prompt['query_set'] else prompt['str_var'])} = ")
    # map char offset to token index
    enc = tok.encode(text)
    for i, (a, b) in enumerate(enc.offsets):
        if a <= char_at < b:
            return ids, i
    # fallback: last token containing the needle's first char
    pos = [i for i, t in enumerate(ids) if needle[0] in tok.decode([t])]
    return ids, pos[-1]


# ---------------- head finding ----------------
@torch.no_grad()
def find_top_head(model, config, tok, prompts, task_score_fn, batch=64):
    base = task_score_fn(model, tok, prompts).mean()
    drops = {}
    for L in range(config.n_layer):
        for H in range(config.n_head):
            sl = slice(H * config.d_head, (H + 1) * config.d_head)

            def zero_head(t, sl=sl):
                t = t.clone()
                t[..., sl] = 0
                return t

            with hook_recorder(regex="$^", interventions={f"{L}.attn.y": zero_head}):
                s = task_score_fn(model, tok, prompts).mean()
            drops[(L, H)] = float(base - s)
    ranked = sorted(drops.items(), key=lambda kv: -kv[1])
    return float(base), ranked


# ---------------- FRA (neuron basis + SAE basis share this) ----------------
def head_qk_weights(model, config, L, H):
    """W_q,h [d_model,dh], W_k,h, b_q [dh], b_k [dh] from fused c_attn; ln_1 scale folded out."""
    W = model.transformer.h[L].attn.c_attn.weight  # [3*nh*dh, d_model]
    b = model.transformer.h[L].attn.c_attn.bias
    dh, nh = config.d_head, config.n_head
    Wq = W[H * dh : (H + 1) * dh, :].T.float().detach()
    Wk = W[nh * dh + H * dh : nh * dh + (H + 1) * dh, :].T.float().detach()
    bq = b[H * dh : (H + 1) * dh].float().detach()
    bk = b[nh * dh + H * dh : nh * dh + (H + 1) * dh].float().detach()
    return Wq, Wk, bq, bk


def omega_from_decoder(Wd_q, Wd_k, Wq, Wk, bq, bk, scale):
    """omega'[(F+1),(F+1)] incl. bias channel: row/col F = the constant contribution.

    Wd_q/Wd_k: [F, d_model] decoder (Identity for neuron basis), or includes b_dec via caller.
    """
    Q = Wd_q @ Wq  # [F, dh]
    K = Wd_k @ Wk
    Fq = torch.cat([Q, bq.unsqueeze(0)], 0)   # [F+1, dh]
    Fk = torch.cat([K, bk.unsqueeze(0)], 0)
    return (Fq @ Fk.T) * scale                # [F+1, F+1]


@torch.no_grad()
def collect_act_in(model, config, L, ids):
    with hook_recorder(regex=f"^{L}\\.attn\\.(act_in|q|k)$") as rec:
        model(ids)
    return (rec[f"{L}.attn.act_in"].float(), rec[f"{L}.attn.q"].float(), rec[f"{L}.attn.k"].float())


def append_bias_channel(U):
    return torch.cat([U, torch.ones_like(U[..., :1])], dim=-1)


@torch.no_grad()
def fra_recon_r2(U1, qh, kh, omega, lens):
    """R2 of FRA recon vs true att over valid causal pairs; U1 has bias channel."""
    r2s = []
    for b in range(U1.shape[0]):
        T = int(lens[b])
        S = (U1[b, :T] @ omega) @ U1[b, :T].T
        A = qh[b, :T] @ kh[b, :T].T * (1.0 / math.sqrt(qh.shape[-1]))
        m = torch.tril(torch.ones(T, T, device=U1.device)).bool()
        res = ((S - A)[m] ** 2).sum()
        tot = ((A[m] - A[m].mean()) ** 2).sum()
        r2s.append(float(1 - res / max(tot, 1e-9)))
    return float(np.mean(r2s))


def edge_cell_matrix(U1, omega, qp, kp):
    return torch.outer(U1[qp], U1[kp]) * omega  # [F+1, F+1]


def cell_metrics(Ms, topk_set=16):
    """Ms: list of [F+1,F+1] edge matrices (one per prompt). Persistence-style metrics."""
    Fp = Ms[0].shape[0]
    dom, msets, qdoms, kdoms = [], [], [], []
    mass1, mass8, mass32 = [], [], []
    for M in Ms:
        a = M.abs()
        flat = a.flatten()
        tot = float(flat.sum()) + 1e-12
        v, idx = flat.topk(min(topk_set, flat.numel()))
        cells = [(int(i // Fp), int(i % Fp)) for i in idx]
        dom.append(cells[0])
        msets.append(set(cells))
        qdoms.append(cells[0][0])
        kdoms.append(cells[0][1])
        sv = flat.sort(descending=True).values
        mass1.append(float(sv[:1].sum()) / tot)
        mass8.append(float(sv[:8].sum()) / tot)
        mass32.append(float(sv[:32].sum()) / tot)
    n = len(Ms)
    from collections import Counter
    cnt = Counter(dom)
    ranked = cnt.most_common()
    top1_coverage = ranked[0][1] / n
    cum, k90 = 0, 0
    for _, c in ranked:
        cum += c
        k90 += 1
        if cum >= 0.9 * n:
            break
    cov3 = sum(c for _, c in ranked[:3]) / n
    cov5 = sum(c for _, c in ranked[:5]) / n
    # pairwise Jaccard of top-k cell sets (set-level drift)
    rng = random.Random(0)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(min(300, n * n))]
    jac = [len(msets[i] & msets[j]) / max(len(msets[i] | msets[j]), 1)
           for i, j in pairs if i != j]
    qc = Counter(qdoms)
    kc = Counter(kdoms)
    return {
        "n": n,
        "top1_cell": list(ranked[0][0]),
        "top1_coverage": top1_coverage,
        "cov3": cov3, "cov5": cov5,
        "n_cells_for_90": k90,
        "n_distinct_dom": len(ranked),
        "edge_mass_top1": float(np.mean(mass1)),
        "edge_mass_top8": float(np.mean(mass8)),
        "edge_mass_top32": float(np.mean(mass32)),
        "q_dom_modal_cov": qc.most_common(1)[0][1] / n,
        "q_dom_distinct": len(qc),
        "k_dom_modal_cov": kc.most_common(1)[0][1] / n,
        "k_dom_distinct": len(kc),
        "jaccard_topcells": float(np.mean(jac)) if jac else None,
        "cell_hist_top10": [[list(c), int(v)] for c, v in ranked[:10]],
    }


def cofire_set_drift(Us, qps):
    """active-channel co-firing-set drift at the q position: full-set + top-32 Jaccard."""
    full, t32 = [], []
    for U, qp in zip(Us, qps):
        u = U[qp]
        nz = set(torch.nonzero(u).flatten().tolist())
        full.append(nz)
        t32.append(set(u.abs().topk(min(32, u.numel())).indices.tolist()))
    rng = random.Random(1)
    n = len(full)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(300)]
    jf = [len(full[i] & full[j]) / max(len(full[i] | full[j]), 1) for i, j in pairs if i != j]
    jt = [len(t32[i] & t32[j]) / max(len(t32[i] | t32[j]), 1) for i, j in pairs if i != j]
    return {"jaccard_active_full": float(np.mean(jf)), "jaccard_top32": float(np.mean(jt)),
            "mean_active": float(np.mean([len(s) for s in full]))}


def edge_matrix_cosine(Ms):
    n = len(Ms)
    rng = random.Random(2)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(200)]
    cs = []
    for i, j in pairs:
        if i == j:
            continue
        a, b = Ms[i].flatten(), Ms[j].flatten()
        cs.append(float(F.cosine_similarity(a, b, dim=0)))
    return float(np.mean(cs))


# ---------------- causal: patched attention with att-delta ----------------
class AttDelta:
    """Global mutable spec consumed by the patched forward."""
    def __init__(self):
        self.layer = None
        self.head = None
        self.fn = None  # fn(x_after_attnin_sparsity)->[B,T,T] delta added to att[:,head]
        self.fixed = None  # or a fixed [B,T,T] tensor

ATT = AttDelta()


def make_patched_forward(L):
    def fwd(self, x):
        x = self.config.maybe_activation_sparsity(x, "attn_in")
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_head * self.d_head, dim=2)
        k = self.config.maybe_activation_sparsity(k, "attn_k")
        q = self.config.maybe_activation_sparsity(q, "attn_q")
        v = self.config.maybe_activation_sparsity(v, "attn_v")
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale
        if ATT.layer == L:
            if ATT.fn is not None:
                att[:, ATT.head] = att[:, ATT.head] + ATT.fn(x)
            elif ATT.fixed is not None:
                att[:, ATT.head] = att[:, ATT.head] + ATT.fixed[:, :T, :T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
        y = self.resid_dropout(self.c_proj(y))
        y = self.config.maybe_activation_sparsity(y, "attn_out")
        return y
    return fwd


class PatchedAttn:
    def __init__(self, model, L):
        self.attn = model.transformer.h[L].attn
        self.L = L

    def __enter__(self):
        self.orig = self.attn.forward
        self.attn.forward = types.MethodType(make_patched_forward(self.L), self.attn)
        return self

    def __exit__(self, *a):
        self.attn.forward = self.orig
        ATT.layer = ATT.head = ATT.fn = ATT.fixed = None


def cells_delta_fn(cells, U1_cached, omega, weights=None):
    """delta(x)->[B,T,T]: subtract listed cells' contributions, position-invariant.

    Uses cached U1 (act_in + bias) so the delta matches the FRA decomposition exactly.
    """
    def fn(x):
        B, T = U1_cached.shape[0], x.shape[1]
        assert x.shape[0] == B, f"causal batch {x.shape[0]} != cached {B} (run single-batch)"
        d = torch.zeros(B, T, T, device=x.device)
        for (mu, nu) in cells:
            w = omega[mu, nu]
            d -= w * torch.einsum("bq,bk->bqk", U1_cached[:, :T, mu], U1_cached[:, :T, nu])
        return d
    return fn


@torch.no_grad()
def causal_run(model, tok, prompts, task_score_fn, L, cells, U1, omega, head):
    ATT.layer, ATT.head = L, head
    ATT.fn = cells_delta_fn(cells, U1, omega)
    with PatchedAttn(model, L):
        # single batch: the delta closure is aligned with the full prompt list
        s = task_score_fn(model, tok, prompts, batch=len(prompts)).mean()
    return float(s)


def removal(base, cut):
    return (base - cut) / max(base - 0.5, 1e-9)


# ---------------- stage A ----------------
def stage_A(state):
    out = {"models": {}, "config": {"N_QUOTE": N_QUOTE, "N_BIND": N_BIND, "seed": SEED}}
    tok = load_tokenizer()
    rng = random.Random(SEED)
    quotes = gen_quote_prompts(N_QUOTE, rng)
    binds = gen_bind_prompts(N_BIND, rng)
    q_loc, q_hold = quotes[: N_QUOTE // 2], quotes[N_QUOTE // 2 :]
    b_loc, b_hold = binds[: N_BIND // 2], binds[N_BIND // 2 :]
    state["tok"] = tok
    state["prompts"] = {"q_loc": q_loc, "q_hold": q_hold, "b_loc": b_loc, "b_hold": b_hold}

    for mkey in ["sparse", "wsda", "dense"]:
        name = MODELS[mkey]
        model, config = load_ws_model(name)
        macc = {}
        for task, ps, fn in [("quote", quotes, quote_scores), ("bind", binds, bind_scores)]:
            sc = fn(model, tok, ps)
            macc[task] = {"acc": float((sc > 0.5).mean()), "mean_score": float(sc.mean())}
        if mkey == "dense" and macc["quote"]["acc"] < TASK_GATE:
            print(f"[gate] {name} quote acc {macc['quote']['acc']:.2f} < {TASK_GATE}; trying {DENSE_FALLBACK}", flush=True)
            name = DENSE_FALLBACK
            MODELS["dense"] = name
            model, config = load_ws_model(name)
            for task, ps, fn in [("quote", quotes, quote_scores), ("bind", binds, bind_scores)]:
                sc = fn(model, tok, ps)
                macc[task] = {"acc": float((sc > 0.5).mean()), "mean_score": float(sc.mean())}
        mrec = {"name": name, "task_acc": macc,
                "arch": {"n_layer": config.n_layer, "n_head": config.n_head,
                         "d_model": config.d_model, "afrac": config.afrac}}

        # --- head finding on quote (probe = first 64 locate prompts) ---
        base, ranked = find_top_head(model, config, tok, q_loc[:64], quote_scores)
        (L, H), drop = ranked[0]
        mrec["quote_head"] = {"layer": L, "head": H, "drop": drop, "base": base,
                              "top5": [[list(k), float(v)] for k, v in ranked[:5]]}
        print(f"[head] {name} quote head L{L}H{H} drop={drop:.3f} base={base:.3f}", flush=True)
        baseB, rankedB = find_top_head(model, config, tok, b_loc[:48], bind_scores)
        (LB, HB), dropB = rankedB[0]
        mrec["bind_head"] = {"layer": LB, "head": HB, "drop": dropB, "base": baseB,
                             "top5": [[list(k), float(v)] for k, v in rankedB[:5]]}

        # --- neuron-basis FRA on the quote edge ---
        dh = config.d_head
        scale = 1.0 / math.sqrt(dh)
        Wq, Wk, bq, bk = head_qk_weights(model, config, L, H)
        I = torch.eye(config.d_model, device=DEV)
        omega = omega_from_decoder(I, I, Wq, Wk, bq, bk, scale)

        def fra_on(prompts_, kpos_fn):
            toks = [tok.encode(p["text"]).ids for p in prompts_]
            ids, lens = pad_batch(toks)
            U, qh_all, kh_all = collect_act_in(model, config, L, ids)
            qh = qh_all[..., H * dh : (H + 1) * dh]
            kh = kh_all[..., H * dh : (H + 1) * dh]
            U1 = append_bias_channel(U)
            r2 = fra_recon_r2(U1, qh, kh, omega, lens)
            Ms, qps, kps = [], [], []
            for b, p in enumerate(prompts_):
                _, kp = kpos_fn(tok, p)
                qp = int(lens[b]) - 1
                Ms.append(edge_cell_matrix(U1[b], omega, qp, kp).cpu())
                qps.append(qp)
                kps.append(kp)
            return U1, Ms, qps, kps, r2, ids, lens

        U1l, Msl, qpsl, kpsl, r2l, idsl, lensl = fra_on(q_loc, find_kpos_quote)
        U1h, Msh, qpsh, kpsh, r2h, idsh, lensh = fra_on(q_hold, find_kpos_quote)
        cm = cell_metrics(Msl + Msh)
        cm["fra_recon_r2_locate"] = r2l
        cm["fra_recon_r2_holdout"] = r2h
        cm["cofire_qside"] = cofire_set_drift([U1l[b].cpu() for b in range(len(Msl))], qpsl)
        cm["edge_cosine"] = edge_matrix_cosine(Msl + Msh)
        mrec["fra_neuron_quote"] = cm

        # binding-edge FRA (drift metrics only, same head-basis but binding head)
        WqB, WkB, bqB, bkB = head_qk_weights(model, config, LB, HB)
        omegaB = omega_from_decoder(I, I, WqB, WkB, bqB, bkB, scale)

        def fra_on_bind(prompts_):
            toks = [tok.encode(p["text"]).ids for p in prompts_]
            ids, lens = pad_batch(toks)
            U, qh_all, kh_all = collect_act_in(model, config, LB, ids)
            U1 = append_bias_channel(U)
            qh = qh_all[..., HB * dh : (HB + 1) * dh]
            kh = kh_all[..., HB * dh : (HB + 1) * dh]
            r2 = fra_recon_r2(U1, qh, kh, omegaB, lens)
            Ms, qps = [], []
            for b, p in enumerate(prompts_):
                _, kp = find_kpos_bind(tok, p)
                qp = int(lens[b]) - 1
                Ms.append(edge_cell_matrix(U1[b], omegaB, qp, kp).cpu())
                qps.append(qp)
            return Ms, r2

        MsB, r2B = fra_on_bind(b_loc + b_hold)
        cmB = cell_metrics(MsB)
        cmB["fra_recon_r2"] = r2B
        mrec["fra_neuron_bind"] = cmB

        # --- causal: Spearman + recovery on the quote task ---
        mrec["causal_neuron_quote"] = causal_block(
            model, tok, q_loc, q_hold, quote_scores, L, H, omega,
            U1l, Msl, U1h, lensl, lensh, qpsl, kpsl, qpsh, kpsh, state, mkey, "neuron")

        out["models"][mkey] = mrec
        state.setdefault("loaded", {})[mkey] = (model, config)
        state.setdefault("fra_ctx", {})[mkey] = {
            "L": L, "H": H, "omega": omega, "idsl": idsl, "idsh": idsh,
            "lensl": lensl, "lensh": lensh, "qpsl": qpsl, "kpsl": kpsl,
            "qpsh": qpsh, "kpsh": kpsh}
        save_stage("A", out)
    state["stageA"] = out
    return out


def spearman(a, b):
    """rank correlation without scipy (Pearson of average ranks)."""
    def ranks(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        r = np.empty_like(x)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


def causal_block(model, tok, p_loc, p_hold, score_fn, L, H, omega,
                 U1l, Msl, U1h, lensl, lensh, qpsl, kpsl, qpsh, kpsh, state, mkey, basis):
    """rank cells on LOCATE (FRA score + causal effect), evaluate recovery on HOLDOUT."""
    n = len(Msl)
    Fp = Msl[0].shape[0]
    # FRA score s = |mean signed edge value| across locate prompts
    Mstack = torch.stack([m for m in Msl])             # [n, F+1, F+1]
    smat = Mstack.mean(0).abs()
    active = (Mstack.abs() > 1e-8).float().mean(0) >= 0.3
    flat_idx = torch.nonzero(active.flatten()).flatten()
    s_flat = smat.flatten()
    top_idx = s_flat.argsort(descending=True)[:N_CAUSAL_TOP].tolist()
    rng = random.Random(SEED + 7)
    pool = [int(i) for i in flat_idx.tolist() if i not in set(top_idx)]
    rand_idx = rng.sample(pool, min(N_CAUSAL_RAND, len(pool)))
    union = top_idx + rand_idx
    cells = [(int(i // Fp), int(i % Fp)) for i in union]

    base_loc = float(score_fn(model, tok, p_loc).mean())
    base_hold = float(score_fn(model, tok, p_hold).mean())

    # VALIDATION GATE: patched forward with zero delta must reproduce the clean score.
    zval = causal_run(model, tok, p_loc, score_fn, L, [], U1l, omega, H)
    assert abs(zval - base_loc) < 1e-3, f"patched-forward mismatch: {zval} vs {base_loc}"

    c_eff, s_val = [], []
    t0 = time.time()
    for ci, cell in enumerate(cells):
        cut = causal_run(model, tok, p_loc, score_fn, L, [cell], U1l, omega, H)
        c_eff.append(removal(base_loc, cut))
        s_val.append(float(s_flat[union[ci]]))
        if ci % 50 == 49:
            print(f"[causal/{mkey}/{basis}] {ci+1}/{len(cells)} cells {time.time()-t0:.0f}s", flush=True)
    sp = spearman(s_val, c_eff)

    # recovery(k) on holdout: FRA-top-k vs causal-top-k
    order_s = np.argsort(-np.array(s_val))
    order_c = np.argsort(-np.array(c_eff))
    rec = {}
    for k in (1, 3, 10):
        cs = [cells[i] for i in order_s[:k]]
        cc = [cells[i] for i in order_c[:k]]
        cut_s = causal_run(model, tok, p_hold, score_fn, L, cs, U1h, omega, H)
        cut_c = causal_run(model, tok, p_hold, score_fn, L, cc, U1h, omega, H)
        rem_s, rem_c = removal(base_hold, cut_s), removal(base_hold, cut_c)
        rec[f"k{k}"] = {"rem_fra": rem_s, "rem_causal": rem_c,
                        "recovery": rem_s / rem_c if abs(rem_c) > 1e-6 else None}
    # oracle ceilings on holdout
    Bh, Th = U1h.shape[0], U1h.shape[1]
    fixed = torch.zeros(Bh, Th, Th, device=DEV)
    for b in range(Bh):
        fixed[b, qpsh[b], kpsh[b]] = -1e9
    ATT.layer, ATT.head, ATT.fixed = L, H, fixed
    with PatchedAttn(model, L):
        cut_mask = float(score_fn(model, tok, p_hold, batch=len(p_hold)).mean())
    orc_mask = removal(base_hold, cut_mask)

    return {"base_locate": base_loc, "base_holdout": base_hold,
            "n_cells_tested": len(cells), "spearman_s_c": sp,
            "recovery": rec, "oracle_edge_mask_rem": orc_mask,
            "top_causal_cells": [[list(cells[i]), float(c_eff[i])] for i in order_c[:5]],
            "top_fra_cells": [[list(cells[i]), float(s_val[i])] for i in order_s[:5]]}


# ---------------- stage B: SAE training ----------------
def python_token_stream(tok, n_tokens):
    """token windows (128) from a public python corpus."""
    from datasets import load_dataset
    sources = [("codeparrot/codeparrot-clean-valid", "content"),
               ("flytech/python-codes-25k", "output"),
               ("mbpp", "code")]
    windows = []
    need = n_tokens // 128 + 1
    for ds_name, field in sources:
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            for doc in ds:
                ids = tok.encode(doc[field]).ids
                for s in range(0, len(ids) - 128, 128):
                    windows.append(ids[s : s + 128])
                    if len(windows) >= need:
                        return windows
            if len(windows) >= need // 2:
                return windows
        except Exception as e:
            print(f"[corpus] {ds_name} failed: {e}", flush=True)
    if not windows:
        raise RuntimeError("no corpus available")
    return windows


@torch.no_grad()
def collect_acts(model, config, L, windows, batch=64):
    chunks = []
    for s in range(0, len(windows), batch):
        ids = torch.tensor(windows[s : s + batch], dtype=torch.long, device=DEV)
        with hook_recorder(regex=f"^{L}\\.attn\\.act_in$") as rec:
            model(ids)
        chunks.append(rec[f"{L}.attn.act_in"].reshape(-1, config.d_model).half().cpu())
    return torch.cat(chunks)


def train_sae(acts_train, acts_eval, d_in, tag):
    torch.manual_seed(SEED)  # identical init across models
    sae = TopKSAE(d_in=d_in, d_sae=SAE_D, k=SAE_K).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    Xtr = acts_train.float().to(DEV)
    n = Xtr.shape[0]
    hist = []
    for step in range(SAE_STEPS):
        idx = torch.randint(0, n, (SAE_BS,), device=DEV)
        x = Xtr[idx]
        xh, z = sae(x)
        loss = ((x - xh) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % 1000 == 999:
            fvu = eval_sae_fvu(sae, acts_eval)
            hist.append({"step": step + 1, "train_mse": float(loss), "eval_fvu": fvu})
            print(f"[sae/{tag}] step {step+1} mse={float(loss):.4f} eval_fvu={fvu:.4f}", flush=True)
    metrics = eval_sae_full(sae, acts_eval)
    metrics["hist"] = hist
    return sae, metrics


@torch.no_grad()
def eval_sae_fvu(sae, acts_eval, cap=100000):
    X = acts_eval[:cap].float().to(DEV)
    xh, z = sae(X)
    mse = ((X - xh) ** 2).sum(-1).mean()
    var = ((X - X.mean(0)) ** 2).sum(-1).mean()
    return float(mse / var)


@torch.no_grad()
def eval_sae_full(sae, acts_eval, cap=200000):
    X = acts_eval[:cap].float().to(DEV)
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=DEV)
    mse_sum, var_sum, n = 0.0, 0.0, 0
    mu = X.mean(0)
    zs_max = torch.zeros(sae.d_sae, device=DEV)
    kurt_acc = torch.zeros(sae.d_sae, device=DEV)
    for s in range(0, X.shape[0], 16384):
        x = X[s : s + 16384]
        xh, z = sae(x)
        fired |= (z > 0).any(0)
        mse_sum += float(((x - xh) ** 2).sum())
        var_sum += float(((x - mu) ** 2).sum())
        zs_max = torch.maximum(zs_max, z.max(0).values)
        n += x.shape[0]
    fvu = mse_sum / max(var_sum, 1e-9)
    l0_eff = None
    xh, z = sae(X[:20000])
    l0_eff = float((z > 0).float().sum(-1).mean())
    return {"fvu": fvu, "r2": 1 - fvu, "dead_frac": float((~fired).float().mean()),
            "l0_config": SAE_K, "l0_effective": l0_eff}


@torch.no_grad()
def monosemanticity(sae, acts_eval, token_ids, cap=200000, topn=50):
    """token-purity: frac of a feature's top-50 activations sharing the modal token."""
    X = acts_eval[:cap].float().to(DEV)
    tids = token_ids[:cap]
    Z = []
    for s in range(0, X.shape[0], 16384):
        _, z = sae(X[s : s + 16384])
        Z.append(z.cpu())
    Z = torch.cat(Z)
    purs = []
    fired = (Z > 0).any(0)
    alive = torch.nonzero(fired).flatten().tolist()
    sample = alive if len(alive) <= 512 else random.Random(3).sample(alive, 512)
    for f in sample:
        zf = Z[:, f]
        k = min(topn, int((zf > 0).sum()))
        if k < 5:
            continue
        top = zf.topk(k).indices
        toks = tids[top].tolist()
        from collections import Counter
        purs.append(Counter(toks).most_common(1)[0][1] / k)
    return {"token_purity_mean": float(np.mean(purs)) if purs else None,
            "n_features_scored": len(purs)}


def stage_B(state):
    out = {"sae_config": {"d_sae": SAE_D, "k": SAE_K, "steps": SAE_STEPS, "bs": SAE_BS,
                          "lr": SAE_LR, "train_tok": N_TRAIN_TOK, "eval_tok": N_EVAL_TOK}}
    tok = state.get("tok") or load_tokenizer()
    windows = python_token_stream(tok, N_TRAIN_TOK + N_EVAL_TOK)
    rng = random.Random(SEED + 1)
    rng.shuffle(windows)
    n_eval_w = N_EVAL_TOK // 128
    eval_w, train_w = windows[:n_eval_w], windows[n_eval_w:]
    eval_token_ids = torch.tensor([t for w in eval_w for t in w])
    out["n_windows"] = {"train": len(train_w), "eval": len(eval_w)}

    for mkey in ["sparse", "wsda", "dense"]:
        model, config = state["loaded"][mkey]
        L = state["fra_ctx"][mkey]["L"]
        t0 = time.time()
        acts_tr = collect_acts(model, config, L, train_w)
        acts_ev = collect_acts(model, config, L, eval_w)
        print(f"[acts/{mkey}] L{L} train {acts_tr.shape} eval {acts_ev.shape} {time.time()-t0:.0f}s", flush=True)
        # input stats (substrate characterization)
        Xs = acts_ev[:50000].float()
        out.setdefault("substrate", {})[mkey] = {
            "frac_nonzero": float((Xs != 0).float().mean()),
            "mean_norm": float(Xs.norm(dim=-1).mean()),
            "kurtosis_mean": float((((Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)) ** 4).mean()),
        }
        sae, met = train_sae(acts_tr, acts_ev, config.d_model, mkey)
        met.update(monosemanticity(sae, acts_ev, eval_token_ids))
        out.setdefault("sae", {})[mkey] = met
        sp = os.path.join(OUT_DIR, f"sae_{mkey}.pt")
        torch.save({"state_dict": sae.state_dict(), "d_in": config.d_model,
                    "d_sae": SAE_D, "k": SAE_K, "layer": L, "model": MODELS[mkey]}, sp)
        upload(sp)
        state.setdefault("saes", {})[mkey] = sae
        save_stage("B", out)
    state["stageB"] = out
    return out


# ---------------- stage C: SAE-basis FRA ----------------
def stage_C(state):
    out = {"models": {}}
    tok = state["tok"]
    P = state["prompts"]
    for mkey in ["sparse", "wsda", "dense"]:
        model, config = state["loaded"][mkey]
        sae = state["saes"][mkey]
        ctx = state["fra_ctx"][mkey]
        L, H = ctx["L"], ctx["H"]
        dh = config.d_head
        scale = 1.0 / math.sqrt(dh)
        Wq, Wk, bq, bk = head_qk_weights(model, config, L, H)
        Wd = sae.W_dec.float()                       # [d_sae, d_model]
        # bias channel = model bias + b_dec contribution
        bq_eff = bq + sae.b_dec.float() @ Wq
        bk_eff = bk + sae.b_dec.float() @ Wk
        omega_s = omega_from_decoder(Wd, Wd, Wq, Wk, bq_eff, bk_eff, scale)

        def sae_fra(ids, lens, qps, kps):
            U, qh_all, kh_all = collect_act_in(model, config, L, ids)
            Z = []
            for b in range(U.shape[0]):
                Z.append(sae.encode(U[b]))
            Z = torch.stack(Z)
            Z1 = append_bias_channel(Z)
            qh = qh_all[..., H * dh : (H + 1) * dh]
            kh = kh_all[..., H * dh : (H + 1) * dh]
            r2 = fra_recon_r2(Z1, qh, kh, omega_s, lens)
            Ms = [edge_cell_matrix(Z1[b], omega_s, qps[b], kps[b]).cpu()
                  for b in range(Z1.shape[0])]
            return Z1, Ms, r2

        Z1l, Msl, r2l = sae_fra(ctx["idsl"], ctx["lensl"], ctx["qpsl"], ctx["kpsl"])
        Z1h, Msh, r2h = sae_fra(ctx["idsh"], ctx["lensh"], ctx["qpsh"], ctx["kpsh"])
        cm = cell_metrics(Msl + Msh)
        cm["fra_recon_r2_locate"] = r2l
        cm["fra_recon_r2_holdout"] = r2h
        cm["cofire_qside"] = cofire_set_drift([Z1l[b].cpu() for b in range(Z1l.shape[0])], ctx["qpsl"])
        cm["edge_cosine"] = edge_matrix_cosine(Msl + Msh)
        mrec = {"fra_sae_quote": cm}

        mrec["causal_sae_quote"] = causal_block(
            model, tok, P["q_loc"], P["q_hold"], quote_scores, L, H, omega_s,
            Z1l, Msl, Z1h, ctx["lensl"], ctx["lensh"], ctx["qpsl"], ctx["kpsl"],
            ctx["qpsh"], ctx["kpsh"], state, mkey, "sae")
        out["models"][mkey] = mrec
        save_stage("C", out)
    state["stageC"] = out
    return out


# ---------------- main ----------------
def main():
    state = {}
    t0 = time.time()
    if "A" in STAGES:
        stage_A(state)
        print(f"[main] stage A done {time.time()-t0:.0f}s", flush=True)
    if "B" in STAGES:
        stage_B(state)
        print(f"[main] stage B done {time.time()-t0:.0f}s", flush=True)
    if "C" in STAGES:
        stage_C(state)
        print(f"[main] stage C done {time.time()-t0:.0f}s", flush=True)
    # combined
    comb = {k: state.get(f"stage{k}") for k in "ABC"}
    p = os.path.join(OUT_DIR, "ws_combined.json")
    with open(p, "w") as f:
        json.dump(comb, f, indent=1, default=float)
    upload(p)
    print("[main] ALL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        tp = os.path.join(OUT_DIR, "ws_traceback.txt")
        with open(tp, "w") as f:
            f.write(tb)
        upload(tp)
        raise
