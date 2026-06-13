#!/usr/bin/env python3
"""Phase F — FEASIBILITY GATE for the in-context backdoor on weight-sparse code models.

Question: can the released OpenAI circuit-sparsity 1x models follow an IN-CONTEXT
mapping at all? (They are tiny, Python-only; they may not.) We run a battery of
in-context association formats, easiest first, and apply the PRE-REGISTERED gate:

  GATE = some mapping FORMAT completed at >= 0.6 top-1 (full-vocab argmax == target)
         on >= 20 held-out instantiations (novel A,B token pairs), at n_demos >= 2
         (a genuine in-context mapping, not a 1-shot local copy).

If NOTHING passes on ANY ladder model -> blocked-by-capability (a valid bound).

Battery (all reported; gate fires if ANY format passes on ANY model):
  RAW_INDUCTION : "A B " * k + "A "         -> P(B)   (pure induction-head test)
  ARROW_COMMENT : "# A -> B\n" * k + "# A ->"-> P(B)  (comment-arrow convention)
  ASSIGN        : "A = B\n" * k + "A ="      -> P(B)   (assignment / alias convention)
  COLON_MAP     : "A: B\n" * k + "A:"        -> P(B)   (dict-like mapping)

Models (OpenAI circuit-sparsity, Gao et al. 2025, arXiv 2511.13653):
  sparse = csp_sweep1_1x_3.7Mnonzero_afrac0.250  (weight+act sparse, 8L)
  wsda   = csp_sweep1_1x_3.7Mnonzero_afrac1.000  (byte-identical, dense acts, 8L)
  dense  = dense1_1x (fully dense, 4L; depth confound flagged)

Resumable (per-model checkpoint), partial-upload, traceback-upload.
HF artifacts: dmanningcoe/fra-phase1-steering-data : fra_ws_backdoor/results/
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csp_vendor_gpt as G

# ---------------- config ----------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", "0"))
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/out")
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
HF_PREFIX = "fra_ws_backdoor/results"
BLOB = "https://openaipublic.blob.core.windows.net/circuit-sparsity/models"

MODELS = {
    "sparse": "csp_sweep1_1x_3.7Mnonzero_afrac0.250",
    "wsda": "csp_sweep1_1x_3.7Mnonzero_afrac1.000",
    "dense": "dense1_1x",
}
MODEL_ORDER = os.environ.get("MODEL_ORDER", "sparse,wsda,dense").split(",")
N_PAIRS = int(os.environ.get("N_PAIRS", "24"))          # held-out instantiations / cell
K_DEMOS = [int(x) for x in os.environ.get("K_DEMOS", "1,2,3,4").split(",")]
GATE_ACC = float(os.environ.get("GATE_ACC", "0.60"))
GATE_NMIN = int(os.environ.get("GATE_NMIN", "20"))
LOCAL_SMOKE = os.environ.get("LOCAL_SMOKE", "0") == "1"

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------- HF up/down ----------------
def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def upload(path, name=None):
    if LOCAL_SMOKE:
        return False
    name = name or os.path.basename(path)
    for i in range(5):
        try:
            hf_api().upload_file(path_or_fileobj=path, path_in_repo=f"{HF_PREFIX}/{name}",
                                 repo_id=HF_REPO, repo_type="dataset")
            return True
        except Exception as e:
            print(f"[upload] {name} attempt {i}: {e}", flush=True)
            time.sleep(20 * (i + 1))
    return False


# ---------------- model loading (reused from B1 ws_pod) ----------------
def fetch_url(url, binary=False):
    for i in range(6):
        try:
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception as e:
            print(f"[fetch] {url} attempt {i}: {e}", flush=True)
            time.sleep(15 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}")


def load_ws_model(name):
    cfg = json.loads(fetch_url(f"{BLOB}/{name}/beeg_config.json"))
    if "n_mlp" in cfg:
        cfg["d_mlp"] = cfg.pop("n_mlp")
    fields = {f.name for f in dataclasses.fields(G.GPTConfig)}
    cfg = {k: v for k, v in cfg.items() if k in fields}
    cfg["flash"] = False
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
    bad_missing = [m for m in missing
                   if not (m.startswith("transformer.h.") and m.endswith(".attn.bias"))]
    assert not unexpected, f"{name} unexpected keys: {unexpected[:8]}"
    assert not bad_missing, f"{name} missing keys: {bad_missing[:8]}"
    model.eval().to(DEV)
    print(f"[load] {name}: n_layer={config.n_layer} n_head={config.n_head} "
          f"d_model={config.d_model} afrac={config.afrac} vocab={config.vocab_size}", flush=True)
    return model, config


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    p = hf_hub_download("openai/circuit-sparsity", "tokenizer.json")
    return Tokenizer.from_file(p)


def pad_batch(tok_lists, pad_id=0):
    T = max(len(t) for t in tok_lists)
    ids = torch.full((len(tok_lists), T), pad_id, dtype=torch.long)
    for i, t in enumerate(tok_lists):
        ids[i, : len(t)] = torch.tensor(t)
    lens = torch.tensor([len(t) for t in tok_lists])
    return ids.to(DEV), lens.to(DEV)


# ---------------- identifier pool (single-token words) ----------------
CANDIDATES = [
    "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "n", "m",
    "val", "name", "out", "data", "msg", "path", "key", "item", "res", "tmp", "obj",
    "arg", "row", "txt", "buf", "tag", "src", "dst", "node", "edge", "list", "dict",
    "set", "map", "idx", "num", "str", "char", "word", "line", "file", "code", "text",
    "head", "tail", "left", "right", "count", "total", "index", "value", "result",
    "input", "output", "start", "end", "size", "flag", "mode", "type", "func", "self",
    "foo", "bar", "baz", "qux", "alpha", "beta", "gamma", "delta", "zeta", "omega",
    "red", "blue", "green", "gold", "gray", "pink", "cyan", "lime", "navy", "teal",
]


def build_pool(tok, sep):
    """words that, as ' {sep}{w}', begin a single stable token at the w boundary."""
    pool = []
    for w in CANDIDATES:
        # render the way it appears mid-line; require a clean token boundary at w[0]
        probe = f"q{sep}{w}"
        enc = tok.encode(probe)
        idx_w = len(f"q{sep}")
        try:
            j = next(i for i, (a, bb) in enumerate(enc.offsets) if a <= idx_w < bb)
        except StopIteration:
            continue
        # the token covering w[0] must render to (optionally space-led) w-prefix
        dec = tok.decode([enc.ids[j]])
        if dec.strip() == w or (dec.strip() and w.startswith(dec.strip())):
            pool.append(w)
    # dedupe by the actual continuation token to avoid collisions
    seen, clean = set(), []
    for w in pool:
        enc = tok.encode(f"q{sep}{w}")
        idx_w = len(f"q{sep}")
        j = next(i for i, (a, bb) in enumerate(enc.offsets) if a <= idx_w < bb)
        tid = enc.ids[j]
        if tid not in seen:
            seen.add(tid)
            clean.append(w)
    return clean


# ---------------- probe construction ----------------
FORMATS = {
    "RAW_INDUCTION": {"occ": "{A} {B} ", "query": "{A} ", "sep": " "},
    "ARROW_COMMENT": {"occ": "# {A} -> {B}\n", "query": "# {A} -> ", "sep": " "},
    "ASSIGN": {"occ": "{A} = {B}\n", "query": "{A} = ", "sep": " "},
    "COLON_MAP": {"occ": "{A}: {B}\n", "query": "{A}: ", "sep": " "},
}


def make_probe(tok, fmt, A, B, n_demos):
    """returns (prompt_ids, target_id, ok) for: n_demos full (A,B) demos + query A -> B."""
    occ = fmt["occ"].format(A=A, B=B)
    query = fmt["query"].format(A=A)
    demos = occ * n_demos
    full = demos + query + B
    enc = tok.encode(full)
    idx_B = len(demos) + len(query)               # first char of the queried B
    try:
        j = next(i for i, (a, bb) in enumerate(enc.offsets) if a <= idx_B < bb)
    except StopIteration:
        return None, None, False
    prompt_ids = enc.ids[:j]
    target_id = enc.ids[j]
    dec = tok.decode([target_id]).strip()
    ok = bool(dec) and (dec == B or B.startswith(dec))
    if len(prompt_ids) < 2:
        ok = False
    return prompt_ids, target_id, ok


@torch.no_grad()
def run_format(model, tok, fmt, pool, n_demos, n_pairs, rng):
    """top-1 (full-vocab argmax==target), P(target), copy-rate (argmax==A token), rank-acc."""
    rows, targets, a_tokens = [], [], []
    used = 0
    tries = 0
    while used < n_pairs and tries < n_pairs * 20:
        tries += 1
        A, B = rng.sample(pool, 2)
        pi, tid, ok = make_probe(tok, fmt, A, B, n_demos)
        if not ok:
            continue
        # token id for A (as it appears at the query) -> copy-bias control
        aenc = tok.encode(fmt["query"].format(A=A))
        a_first = aenc.ids[0] if aenc.ids else -1
        rows.append(pi)
        targets.append(tid)
        a_tokens.append(a_first)
        used += 1
    if used < 5:
        return {"n": used, "skipped": True}
    ids, lens = pad_batch(rows)
    top1, ptgt, copyrate = [], [], []
    B0 = 0
    BATCH = 128
    for s in range(0, len(rows), BATCH):
        chunk_ids, chunk_lens = ids[s:s + BATCH], lens[s:s + BATCH]
        logits, _, _ = model(chunk_ids)
        for b in range(chunk_ids.shape[0]):
            gi = s + b
            lg = logits[b, chunk_lens[b] - 1].float()
            pr = F.softmax(lg, dim=-1)
            am = int(lg.argmax())
            top1.append(1.0 if am == targets[gi] else 0.0)
            ptgt.append(float(pr[targets[gi]]))
            copyrate.append(1.0 if am == a_tokens[gi] else 0.0)
    return {
        "n": used,
        "top1_acc": float(np.mean(top1)),
        "p_target_mean": float(np.mean(ptgt)),
        "copy_rate": float(np.mean(copyrate)),
        "skipped": False,
    }


def eval_model(mkey, tok):
    model, config = load_ws_model(MODELS[mkey])
    rng = random.Random(SEED + hash(mkey) % 1000)
    res = {"model": MODELS[mkey], "arch": {"n_layer": config.n_layer, "afrac": config.afrac},
           "formats": {}}
    best = {"top1_acc": -1.0}
    for fname, fmt in FORMATS.items():
        pool = build_pool(tok, fmt["sep"])
        res["formats"][fname] = {"pool_size": len(pool), "by_k": {}}
        for k in K_DEMOS:
            r = run_format(model, tok, fmt, pool, k, N_PAIRS, random.Random(SEED + k))
            res["formats"][fname]["by_k"][str(k)] = r
            tag = f"{mkey}/{fname}/k{k}"
            if r.get("skipped"):
                print(f"[feas] {tag}: SKIP (n={r['n']})", flush=True)
                continue
            print(f"[feas] {tag}: top1={r['top1_acc']:.2f} P(tgt)={r['p_target_mean']:.3f} "
                  f"copy={r['copy_rate']:.2f} n={r['n']}", flush=True)
            # gate candidate: k>=2 (genuine in-context, not 1-shot copy), n>=GATE_NMIN
            if k >= 2 and r["n"] >= GATE_NMIN and r["top1_acc"] > best["top1_acc"]:
                best = {"format": fname, "k": k, "top1_acc": r["top1_acc"],
                        "p_target": r["p_target_mean"], "n": r["n"]}
    res["best_gateable"] = best
    res["passes_gate"] = bool(best.get("top1_acc", -1) >= GATE_ACC)
    del model
    return res


def main():
    tok = load_tokenizer()
    out_path = os.path.join(OUT_DIR, "feas_results.json")
    out = {"config": {"seed": SEED, "n_pairs": N_PAIRS, "k_demos": K_DEMOS,
                      "gate_acc": GATE_ACC, "gate_nmin": GATE_NMIN, "dev": DEV},
           "models": {}}
    if os.path.exists(out_path):
        try:
            out = json.load(open(out_path))
        except Exception:
            pass
    for mkey in MODEL_ORDER:
        if mkey in out.get("models", {}):
            print(f"[feas] {mkey} already done, skip", flush=True)
            continue
        out.setdefault("models", {})[mkey] = eval_model(mkey, tok)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1, default=float)
        upload(out_path)
        print(f"[feas] {mkey} -> passes_gate={out['models'][mkey]['passes_gate']} "
              f"best={out['models'][mkey]['best_gateable']}", flush=True)
    # overall verdict
    any_pass = any(m.get("passes_gate") for m in out["models"].values())
    out["verdict"] = "FEASIBLE" if any_pass else "BLOCKED_BY_CAPABILITY"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    upload(out_path)
    print(f"[feas] VERDICT = {out['verdict']}", flush=True)
    for mkey, m in out["models"].items():
        print(f"  {mkey}: gate={m['passes_gate']} best={m['best_gateable']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        tp = os.path.join(OUT_DIR, "feas_traceback.txt")
        with open(tp, "w") as f:
            f.write(tb)
        upload(tp)
        raise
