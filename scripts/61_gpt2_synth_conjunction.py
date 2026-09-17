"""GPT-2 synthetic BIGRAM conjunction feasibility (CPU). Forward passes only.

GPT-2-small can't bind the natural-text password conjunction (scripts/59) but does synthetic single-
token induction strongly (scripts/40, ASR 0.97-0.99). Question: does it do a BIGRAM conjunction --
copy payload P only after the pair (A B), where each token alone is ambiguous (B also follows D->S; A
also precedes C->Q)? If P(P | ...A B) >> P(P | ...D B) and >> P(P | ...A C), that is a genuine AND
(query-content A x key-content B) and the whole B1 removal can run locally on GPT-2.

Construction (in-context, induction-style): random-token filler with three planted triples
  A B P   |   D B S   |   A C Q
placed once, then a query prefix "... A B" (second occurrence of A B) -> predict next.
Single-token induction on B is ambiguous (B->? is P after A, S after D); only using BOTH A and B picks P.
"""
import os, torch, numpy as np
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
NS = int(os.environ.get("NSEED", "40")); REP = int(os.environ.get("REP","1")); dev = "cpu"
m = HookedTransformer.from_pretrained("gpt2", device=dev); m.eval()
print("[model] gpt2 cpu", flush=True)
V = m.cfg.d_vocab
def trial(seed):
    g = torch.Generator().manual_seed(seed)
    # six distinct content tokens A,B,C,D and payloads P,S,Q from a safe id range
    toks = (torch.randperm(30000, generator=g)[:7] + 1500).tolist()
    A, B, C, D, P, S, Q = toks
    fill = (torch.randperm(20000, generator=g)[:40] + 22000).tolist()
    # place three rules separated by filler, REP times, then query A B
    seq = [m.tokenizer.bos_token_id]
    for rep in range(REP):
        seq += [fill[(rep*3+0) % len(fill)], A, B, P, fill[(rep*3+1) % len(fill)], D, B, S, fill[(rep*3+2) % len(fill)], A, C, Q]
    def pnext(prefix, tid):
        ids = seq + prefix
        lg = m(torch.tensor(ids, device=dev).unsqueeze(0))[0][-1].float()
        return torch.softmax(lg, -1)[tid].item()
    # queries: the pair A B, and each token with the OTHER partner (ambiguous marginals)
    pAB = pnext([A, B], P)      # pair -> want high P(P)
    pDB = pnext([D, B], P)      # B with wrong partner D -> want low P(P) (should favour S)
    pAC = pnext([A, C], P)      # A with wrong partner C -> want low P(P) (should favour Q)
    return pAB, pDB, pAC
res = np.array([trial(s) for s in range(NS)])
mAB, mDB, mAC = res.mean(0)
print(f"\n#### GPT-2 synthetic bigram conjunction (mean over {NS} random trials) ####", flush=True)
print(f"  P(P | A B)  = {mAB:.3f}   <- want HIGH (the pair)", flush=True)
print(f"  P(P | D B)  = {mDB:.3f}   <- want LOW  (B, wrong partner)", flush=True)
print(f"  P(P | A C)  = {mAC:.3f}   <- want LOW  (A, wrong partner)", flush=True)
print(f"  ratio pair/max-marginal = {mAB / max(mDB, mAC, 1e-6):.1f}x", flush=True)
print(f"  CONJUNCTION = {mAB >= 0.2 and mDB < 0.1 and mAC < 0.1}", flush=True)
print("\nDONE gpt2_synth_conjunction", flush=True)
