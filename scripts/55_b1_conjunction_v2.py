"""B1 STEP0 v2 -- conjunction via AMBIGUOUS bigrams. Forward passes only. gemma-2-2b base.
Each single token is shared across pairs mapping to DIFFERENT payloads, so neither token alone
determines the payload -- only the pair does. In-context few-shot: show the mappings, then query.
Conjunctive if pair P(payload) is high while each token with a NOVEL partner stays low."""
import os, torch, numpy as np
from transformer_lens import HookedTransformer
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
PAIR=float(os.environ.get("PAIR_THR","0.20")); MARG=float(os.environ.get("MARG_THR","0.10")); NS=int(os.environ.get("NSEED","8"))
m=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); m.eval(); tok=m.tokenizer
print("[dev]",dev,flush=True)
def one(w): return len(tok.encode(w,add_special_tokens=False))==1
def tid(w): return tok.encode(w,add_special_tokens=False)[0]
# each group: list of (A,B,payload) demos sharing tokens; target is demos[0]; novel partners unseen in group
GROUPS=[
 dict(demos=[("red","fox","nine"),("blue","fox","three"),("red","owl","seven")], novelB="cat", novelA="gray"),
 dict(demos=[("iron","gate","four"),("iron","lake","six"),("steel","gate","two")], novelB="pond", novelA="brass"),
 dict(demos=[("blue","moon","eight"),("blue","sun","one"),("pale","moon","five")], novelB="star", novelA="dark"),
]
def Ppay(demos, qa, qb, pay_id, seed):
    g=np.random.default_rng(seed); order=list(g.permutation(len(demos)))
    body="".join(f" {demos[i][0]} {demos[i][1]}: {demos[i][2]}." for i in order)
    text=body+f" {qa} {qb}:"
    ids=[tok.bos_token_id]+tok.encode(text,add_special_tokens=False)
    return torch.softmax(m(torch.tensor(ids,device=dev).unsqueeze(0))[0][-1].float(),-1)[pay_id].item()
def mean(demos,qa,qb,pid): return float(np.mean([Ppay(demos,qa,qb,pid,s) for s in range(NS)]))
print("\n#### v2: ambiguous-bigram conjunction (P of the TARGET payload) ####",flush=True)
print("  target pair should be HIGH; same-token-with-novel-partner should be LOW",flush=True)
for G in GROUPS:
    A,B,pay=G["demos"][0]
    if not all(one(f" {w}") for w in [A,B,pay,G["novelA"],G["novelB"]]):
        print(f"  skip {A}+{B}: some token multitoken",flush=True); continue
    pid=tid(f" {pay}")
    pair =mean(G["demos"],A,B,pid)                 # red fox   -> want high
    aNov =mean(G["demos"],A,G["novelB"],pid)       # red cat   -> A with unseen partner, want low
    bNov =mean(G["demos"],G["novelA"],B,pid)       # gray fox  -> B with unseen partner, want low
    conj = pair>=PAIR and aNov<MARG and bNov<MARG
    print(f"  {A}+{B}->{pay}: pair={pair:.3f}  {A}+NOVEL={aNov:.3f}  NOVEL+{B}={bNov:.3f}  CONJ={conj}",flush=True)
print("\nDONE b1_v2",flush=True)
