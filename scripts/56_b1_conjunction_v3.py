"""B1 STEP0 v3 -- conjunction: STRONG password-recall format + AMBIGUOUS tokens. gemma-2-2b base.
Marries v1(ii)'s strong binding ("password for X Y is Z") with v2's ambiguity (each token shared
across pairs -> different payloads). Want: pair P(payload) HIGH, each token+novel-partner LOW."""
import os, torch, numpy as np
from transformer_lens import HookedTransformer
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
PAIR=float(os.environ.get("PAIR_THR","0.20")); MARG=float(os.environ.get("MARG_THR","0.10")); NS=int(os.environ.get("NSEED","8"))
m=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); m.eval(); tok=m.tokenizer
print("[dev]",dev,flush=True)
FILL=["The weather was mild.","People arrived late.","The library opened early.","A dog waited by the door.",
 "The report took long.","The train was quiet.","She left a note on the desk.","The garden needed water.",
 "Two students argued.","The coffee had gone cold.","Chairs were stacked by the wall.","He forgot his umbrella."]
def one(w): return len(tok.encode(w,add_special_tokens=False))==1
def tid(w): return tok.encode(w,add_special_tokens=False)[0]
GROUPS=[
 dict(demos=[("red","fox","nine"),("blue","fox","three"),("red","owl","seven")], novelB="cat", novelA="gray"),
 dict(demos=[("iron","gate","four"),("iron","lake","six"),("steel","gate","two")], novelB="pond", novelA="brass"),
 dict(demos=[("blue","moon","eight"),("blue","sun","one"),("pale","moon","five")], novelB="star", novelA="dark"),
]
def Ppay(demos, qa, qb, pid, seed):
    g=np.random.default_rng(seed); order=list(g.permutation(len(demos))); fs=[FILL[i] for i in g.permutation(len(FILL))]
    body=" ".join(fs[:3])+"".join(f" The password for {demos[i][0]} {demos[i][1]} is {demos[i][2]}." for i in order)+" "+" ".join(fs[3:6])
    text=body+f" Remember the password for {qa} {qb}:"
    ids=[tok.bos_token_id]+tok.encode(text,add_special_tokens=False)
    return torch.softmax(m(torch.tensor(ids,device=dev).unsqueeze(0))[0][-1].float(),-1)[pid].item()
def mean(demos,qa,qb,pid): return float(np.mean([Ppay(demos,qa,qb,pid,s) for s in range(NS)]))
print("\n#### v3: password-recall + ambiguous bigram (P of TARGET payload) ####",flush=True)
print("  pair should be HIGH; same-token-with-novel-partner should be LOW",flush=True)
for G in GROUPS:
    A,B,pay=G["demos"][0]
    if not all(one(f" {w}") for w in [A,B,pay,G["novelA"],G["novelB"]]):
        print(f"  skip {A}+{B}: some token multitoken",flush=True); continue
    pid=tid(f" {pay}")
    pair=mean(G["demos"],A,B,pid)
    aNov=mean(G["demos"],A,G["novelB"],pid)
    bNov=mean(G["demos"],G["novelA"],B,pid)
    conj=pair>=PAIR and aNov<MARG and bNov<MARG
    print(f"  {A}+{B}->{pay}: pair={pair:.3f}  {A}+NOVEL={aNov:.3f}  NOVEL+{B}={bNov:.3f}  CONJ={conj}",flush=True)
print("\nDONE b1_v3",flush=True)
