"""Attention-only (NO MLP) Mess3 transformer — Phase-0 loss gate + platform.

Adapts sprint phase1/train_p2.py (read-only ref) to attn_only=True at variable
depth. Trains one (config, n_layers, n_heads) cell, computes exact-enumeration CE,
writes out/gate_<tag>.json INCREMENTALLY (network-drop safe).

Usage: train_ao.py <A|B> <n_layers> <n_heads> [--steps N] [--seed S]
Anchors (from sprint phase1/out, cited in ATTN_ONLY.md):
  A (x.15,z+.55): bayes 1.0892030  MLP 1.0894620  constr 1.0895620  unigram 1.0986123
  B (x.50,z-.50): bayes 1.0905314  MLP 1.0908823  constr 1.0908534  unigram 1.0986123
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch
from mess3 import Mess3, enumerate_sequences, prefix_probs

torch.set_num_threads(4)
N_CTX, D_MODEL, BATCH, LR = 10, 64, 128, 1e-4
CFG = {"A": dict(x=0.15, a=0.6), "B": dict(x=0.50, a=0.6)}
ANCHORS = {  # exact-enumeration, from sprint phase1/out/{A,B}
    "A": dict(bayes=1.0892029752, mlp=1.0894620307, constr=1.0895620144, unigram=1.0986122887),
    "B": dict(bayes=1.0905314271, mlp=1.0908823063, constr=1.0908533556, unigram=1.0986122887),
}
OUT = __import__("pathlib").Path(__file__).resolve().parent.parent / "out"


def build_ao(n_heads, n_layers, seed):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(
        n_layers=n_layers, d_model=D_MODEL, n_ctx=N_CTX, d_head=D_MODEL // n_heads,
        n_heads=n_heads, d_vocab=3, d_vocab_out=3, attn_only=True,
        normalization_type="LN", attention_dir="causal", default_prepend_bos=False,
        positional_embedding_type="standard", device="cpu", seed=seed)
    return HookedTransformer(cfg)


@torch.no_grad()
def exact_model_ce(model, m, chunk=4096):
    seqs = enumerate_sequences(N_CTX)
    pp = prefix_probs(m, seqs)
    eta = m.bayes_beliefs(seqs)
    p_true = m.next_token_dist(eta[:, 1:N_CTX])
    w_seq = pp[:, N_CTX]
    ce = np.zeros(N_CTX - 1)
    for i in range(0, seqs.shape[0], chunk):
        toks = torch.tensor(seqs[i:i + chunk])
        logp = torch.log_softmax(model(toks, return_type="logits")[:, :-1], -1).numpy()
        wt = w_seq[i:i + chunk, None]
        ce += -(wt[:, :, None] * p_true[i:i + chunk] * logp).sum(axis=(0, 2))
    return float(ce.mean()), ce.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=list(CFG))
    ap.add_argument("n_layers", type=int)
    ap.add_argument("n_heads", type=int)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tag = f"{args.config}_{args.n_layers}L{args.n_heads}H_s{args.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    jpath = OUT / f"gate_{tag}.json"
    c = CFG[args.config]; anch = ANCHORS[args.config]
    rec = dict(tag=tag, config=args.config, n_layers=args.n_layers, n_heads=args.n_heads,
               steps=args.steps, seed=args.seed, status="running", anchors=anch)
    jpath.write_text(json.dumps(rec, indent=2))

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    m = Mess3(c["x"], c["a"])
    model = build_ao(args.n_heads, args.n_layers, args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(args.seed)
    losses, t0 = [], time.time()
    model.train()
    for step in range(args.steps):
        batch = torch.tensor(m.sample(BATCH, N_CTX, rng))
        loss = model(batch, return_type="loss")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 2000 == 0:
            rec["progress"] = f"{step+1}/{args.steps} loss={np.mean(losses[-200:]):.5f} ({time.time()-t0:.0f}s)"
            jpath.write_text(json.dumps(rec, indent=2))
            print(f"[{tag}] " + rec["progress"], flush=True)
    model.eval()
    ce_mean, ce_pos = exact_model_ce(model, m)
    rec_state = OUT / f"model_{tag}.pt"
    torch.save({"state_dict": model.state_dict(),
                "cfg": dict(n_layers=args.n_layers, n_heads=args.n_heads,
                            x=c["x"], a=c["a"], seed=args.seed)}, rec_state)
    # recovery fraction of the belief-recoverable information
    rec_frac = (anch["unigram"] - ce_mean) / (anch["unigram"] - anch["bayes"])
    rec.update(status="done", model_ce=ce_mean, model_ce_per_pos=ce_pos,
               final_train_loss=float(np.mean(losses[-500:])),
               recovery_frac=rec_frac,
               gap_to_bayes=ce_mean - anch["bayes"],
               gap_to_constrained=ce_mean - anch["constr"],
               vs_mlp=ce_mean - anch["mlp"],
               train_time_s=time.time() - t0, model_path=str(rec_state))
    jpath.write_text(json.dumps(rec, indent=2))
    print(f"[{tag}] DONE ce={ce_mean:.6f} rec={rec_frac:.4f} "
          f"gap_constr={ce_mean-anch['constr']:+.2e} vs_mlp={ce_mean-anch['mlp']:+.2e}",
          flush=True)


if __name__ == "__main__":
    main()
