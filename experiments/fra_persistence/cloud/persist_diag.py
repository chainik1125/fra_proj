"""Diagnostic: for random single-token A->B induction pairs, log the JOINT distribution of
(gate2 mean copy-prob, L5H5 oracle-edge-removal, multi-head oracle-removal) so we can choose
the right locus + thresholds.  Cheap (gpt2-small).  Uploads diag.json to fra_persist/results/.
"""
import os, sys, json, traceback
sys.path.insert(0, "/workspace/code")
OUT = os.environ.get("OUTDIR", ".")
HF_REPO = "dmanningcoe/fra-phase1-steering-data"


def _hf():
    import shutil
    return shutil.which("hf") or shutil.which("huggingface-cli") or "hf"


def upload(p, name):
    import subprocess
    try:
        subprocess.run([_hf(), "upload", HF_REPO, p, f"fra_persist/results/{name}",
                        "--repo-type", "dataset"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=300)
    except Exception:
        pass


def main():
    import torch, numpy as np
    from transformer_lens import HookedTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval()
    tok = model.tokenizer
    L, H = 5, 5
    IND_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]

    def E(s): return tok.encode(s)
    def enc1(s):
        ids = tok.encode(s); return ids[0] if len(ids) == 1 else None
    FILLER = [enc1(w) for w in [" people", " time", " was", " with", " it", " they",
                                " some", " many", " day", " place"] if enc1(w) is not None]
    LC = [enc1(s) for s in [" the", " and", " near", " my", " then"] if enc1(s) is not None]

    def fill_ids(n, seed):
        r = np.random.RandomState(seed)
        return [FILLER[r.randint(len(FILLER))] for _ in range(n)]

    def build_T1(A, B, lc, seed):
        f = fill_ids(10, seed)
        return [A, B] + E(".") + f[:3] + E(".") + [lc, A]

    def make_tt(ids, A, B):
        ids = [tok.bos_token_id] + list(ids)
        if ids[-1] != A: return None
        bpos = [p for p, t in enumerate(ids) if t == B]
        if len(bpos) != 1: return None
        return torch.tensor(ids, device=dev).unsqueeze(0), len(ids), len(ids) - 1, bpos[0]

    def copyprob(tt, q, B, score_delta=None, heads=None):
        if score_delta is None:
            lg = model(tt)[0]
        else:
            seq = tt.shape[1]
            hooks = []
            for (l, h) in (heads or [(L, H)]):
                sd = torch.tensor(score_delta[(l, h)], device=dev, dtype=torch.float32)
                def mk(hh, sdt):
                    def hook(s, hook):
                        s[0, hh, :seq, :seq] = s[0, hh, :seq, :seq] - sdt
                        return s
                    return hook
                hooks.append((f"blocks.{l}.attn.hook_attn_scores", mk(h, sd)))
            lg = model.run_with_hooks(tt, fwd_hooks=hooks)[0]
        return float(torch.softmax(lg[q].float(), -1)[B].item())

    def head_scores(tt, l, h):
        nm = f"blocks.{l}.attn.hook_attn_scores"
        _, c = model.run_with_cache(tt, names_filter=[nm])
        return c[nm][0, h].float().cpu().numpy()

    # random pool
    rand_ids = []
    cand = (torch.randperm(40000, generator=torch.Generator().manual_seed(0))[:1500] + 1000).tolist()
    for t in cand:
        s = tok.decode([t])
        if len(tok.encode(s)) == 1 and s.strip() != "":
            rand_ids.append(t)
        if len(rand_ids) >= 60:
            break

    rows = []
    n = 0
    for ai in range(len(rand_ids)):
        for bi in range(len(rand_ids)):
            if ai == bi: continue
            A, B = rand_ids[ai], rand_ids[bi]
            n += 1
            if n > 300: break
            # gate2: mean copyprob over 5 lc x T1
            cps = []
            tt0 = None
            for lc in LC:
                e = make_tt(build_T1(A, B, lc, 11), A, B)
                if e is None: continue
                tt, seq, q, k = e
                cps.append(copyprob(tt, q, B))
                if tt0 is None: tt0 = (tt, seq, q, k)
            if not cps or tt0 is None: continue
            g2 = float(np.mean(cps))
            if g2 < 0.20:   # only log plausibly-inducing pairs
                continue
            tt, seq, q, k = tt0
            cp0 = copyprob(tt, q, B)
            if cp0 < 1e-6: continue
            # L5H5 oracle
            sc = head_scores(tt, L, H)
            d = {(L, H): np.zeros((seq, seq))}; d[(L, H)][q, k] = sc[q, k] + 20.0
            o_single = 1.0 - copyprob(tt, q, B, score_delta=d, heads=[(L, H)]) / cp0
            # multi-head oracle: zero the induction edge across all IND_HEADS
            dm = {}
            for (l, h) in IND_HEADS:
                scc = head_scores(tt, l, h)
                dm[(l, h)] = np.zeros((seq, seq)); dm[(l, h)][q, k] = scc[q, k] + 20.0
            o_multi = 1.0 - copyprob(tt, q, B, score_delta=dm, heads=IND_HEADS) / cp0
            rows.append(dict(A=tok.decode([A]), B=tok.decode([B]), g2=g2,
                             cp0=cp0, oracle_single=float(o_single), oracle_multi=float(o_multi)))
        if n > 300: break

    rows.sort(key=lambda r: -r["g2"])
    print(f"[diag] {len(rows)} pairs with g2>=0.20", flush=True)
    import numpy as np
    g2a = np.array([r["g2"] for r in rows]); os1 = np.array([r["oracle_single"] for r in rows])
    osm = np.array([r["oracle_multi"] for r in rows])
    print(f"[diag] g2: mean {g2a.mean():.2f} max {g2a.max():.2f} | "
          f"oracle_single: mean {os1.mean():.2f} max {os1.max():.2f} | "
          f"oracle_multi: mean {osm.mean():.2f} max {osm.max():.2f}", flush=True)
    for thr in [0.10, 0.20, 0.40]:
        n_s = int(((g2a >= 0.30) & (os1 >= thr)).sum())
        n_m = int(((g2a >= 0.30) & (osm >= thr)).sum())
        print(f"[diag] g2>=0.30 & oracle>={thr}: single={n_s} multi={n_m}", flush=True)
    print("[diag] top-8 by g2:", flush=True)
    for r in rows[:8]:
        print(f"   {r['A']!r}->{r['B']!r} g2={r['g2']:.2f} o_single={r['oracle_single']:.2f} o_multi={r['oracle_multi']:.2f}", flush=True)
    print("[diag] top-8 by oracle_multi:", flush=True)
    for r in sorted(rows, key=lambda x: -x["oracle_multi"])[:8]:
        print(f"   {r['A']!r}->{r['B']!r} g2={r['g2']:.2f} o_single={r['oracle_single']:.2f} o_multi={r['oracle_multi']:.2f}", flush=True)
    json.dump(dict(rows=rows), open(os.path.join(OUT, "diag.json"), "w"), indent=2, default=float)
    upload(os.path.join(OUT, "diag.json"), "diag.json")
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc(); print("[FATAL]\n" + tb, flush=True)
        open(os.path.join(OUT, "diag_tb.txt"), "w").write(tb); upload(os.path.join(OUT, "diag_tb.txt"), "diag_tb.txt")
        sys.exit(1)
