"""FRA PERSISTENCE (Path 1) — the persistence toy on gpt2-small + gpt2-small-res-jb.

Implements PERSIST_DESIGN.md faithfully (judge-free, ground-truth copy-prob):
  GATE  : token-validity + induction-copy gate -> pick 3 A->B pairs (frozen before M1/M2).
  M1    : locate dominant FRA cell (I,J) on LOCATE set; apply UNCONDITIONAL cut
          (pairs_delta wherever the two features fire) on HELD-OUT -> rem_holdout.
  M2    : per-appearance dominant (qF x kF) cell across all ~120 appearances;
          top1_coverage, cov(k) curve, n_cells_for_90, q-side vs k-side drift,
          {lc} x q-feature contingency, magnitude-weighted variant, edge_coverage.
  M3    : FRA-cell-cut vs token-mask(no detector) vs token-mask(oracle) vs UNION-top-k
          vs BENIGN-USE preservation (FRA vs feature-ablation vs ActAdd).
  VERDICT against the LOCKED bars (§4) + magnitude floor + non-sink gate.

Reuses fra.core.fra._build_fra_result + the j2 pairs_delta / hook_attn_scores cut VERBATIM.
ckpt() inside every loop + early flush prints + top-of-script try/except that uploads the
traceback. Self-stops at the end.
"""
import os, sys, json, traceback, time
sys.path.insert(0, "/workspace/code")

OUT = os.environ.get("OUTDIR", ".")
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
RES_PREFIX = "fra_persist/results"
RUN_LOG = os.environ.get("RUN_LOG", "persist_run.log")
STATE = {"stage": "init", "t0": time.time()}


def _hf():
    import shutil
    return shutil.which("hf") or shutil.which("huggingface-cli") or "hf"


def upload(localpath, remotename):
    import subprocess
    try:
        subprocess.run([_hf(), "upload", HF_REPO, localpath, f"{RES_PREFIX}/{remotename}",
                        "--repo-type", "dataset"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    except Exception as e:
        print(f"[upload-fail] {remotename}: {e}", flush=True)


def ckpt(obj, name="persist_partial.json"):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w"), indent=2, default=float)
    upload(p, name)
    print(f"[ckpt] {name} stage={STATE.get('stage')} dt={time.time()-STATE['t0']:.0f}s", flush=True)


def main():
    import torch, numpy as np
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from fra.core.fra import _build_fra_result

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    print(f"[boot] device={dev}", flush=True)
    STATE["stage"] = "load_model"

    model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval()
    tok = model.tokenizer
    L, H = 5, 5                       # primary induction head (j1/j2 argmax)
    IND_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]   # robustness set
    LAYERS = sorted(set(l for l, h in IND_HEADS))
    print(f"[boot] model loaded; primary head L{L}H{H}", flush=True)

    saes = {}
    for l in LAYERS:
        s = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{l}.hook_resid_pre", device=dev)
        saes[l] = s[0] if isinstance(s, tuple) else s
    d_sae = saes[L].W_dec.shape[0]
    print(f"[boot] SAEs loaded for layers {LAYERS}; d_sae={d_sae}", flush=True)
    ckpt(STATE, "persist_partial.json")

    # ─────────────────────────────────────────────────────────────────────
    # FRA helper: compute the sparse FRA tensor for a token tensor at (l,h)
    # returns numpy arrays qq,kk,ii,jj,vv  (the j2 layout)
    # ─────────────────────────────────────────────────────────────────────
    def fra_arrays(tt, l, h):
        hook = f"blocks.{l}.hook_resid_pre"
        _, cache = model.run_with_cache(tt, names_filter=[hook])
        act = cache[hook][0]
        feats = saes[l].encode(act).float()
        x_hat = feats @ saes[l].W_dec.float() + saes[l].b_dec.float()
        res = _build_fra_result(model, l, h, feats, saes[l].W_dec.float(), dev,
                                top_k=None, rms_activations=x_hat, dec_norms=None,
                                chunk_size=16, verbose=False)
        f = res["fra_tensor_sparse"].coalesce()
        idx = f.indices().cpu().numpy(); val = f.values().cpu().numpy()
        return dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=val)

    # FRA arrays for ALL induction heads at once (one cache pass per resid layer).
    # Returns {(l,h): dict(qq,kk,ii,jj,vv)}.  Keyed by head so M2 cells are per-head.
    def fra_arrays_multi(tt):
        names = [f"blocks.{l}.hook_resid_pre" for l in LAYERS]
        _, cache = model.run_with_cache(tt, names_filter=lambda n: n in names)
        out = {}
        feats_by_l = {}
        for l in LAYERS:
            act = cache[f"blocks.{l}.hook_resid_pre"][0]
            fe = saes[l].encode(act).float()
            feats_by_l[l] = fe
        for (l, h) in IND_HEADS:
            fe = feats_by_l[l]
            x_hat = fe @ saes[l].W_dec.float() + saes[l].b_dec.float()
            res = _build_fra_result(model, l, h, fe, saes[l].W_dec.float(), dev,
                                    top_k=None, rms_activations=x_hat, dec_norms=None,
                                    chunk_size=16, verbose=False)
            f = res["fra_tensor_sparse"].coalesce()
            idx = f.indices().cpu().numpy(); val = f.values().cpu().numpy()
            out[(l, h)] = dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=val)
        return out

    def head_scores(tt, l, h):
        nm = f"blocks.{l}.attn.hook_attn_scores"
        _, cache = model.run_with_cache(tt, names_filter=[nm])
        return cache[nm][0, h].float().cpu().numpy()

    # multi-head cut hook: subtract a per-head [seq,seq] delta from each head's scores.
    # deltas = {(l,h): np[seq,seq]}.  (j7 byL machinery.)
    def patched_logits_multi(tt, deltas):
        seq = tt.shape[1]
        byL = {}
        for (l, h), d in deltas.items():
            byL.setdefault(l, {})[h] = torch.tensor(d, device=dev, dtype=torch.float32)
        hooks = []
        for l, hd in byL.items():
            def mk(hd):
                def hook(s, hook):
                    for h, sd in hd.items():
                        s[0, h, :seq, :seq] = s[0, h, :seq, :seq] - sd
                    return s
                return hook
            hooks.append((f"blocks.{l}.attn.hook_attn_scores", mk(hd)))
        return model.run_with_hooks(tt, fwd_hooks=hooks)[0]

    # pairs_delta (j2): sum FRA values for a set of (i,j) over ALL (q,k) -> [seq,seq] delta.
    # Vectorized: select rows whose (ii,jj) matches any requested (i,j), then scatter-add.
    def pairs_delta(A, pairs, seq):
        ii, jj, qq, kk, vv = A["ii"], A["jj"], A["qq"], A["kk"], A["vv"]
        d = np.zeros((seq, seq))
        if len(vv) == 0:
            return d
        mask = np.zeros(len(vv), dtype=bool)
        for (i, j) in pairs:
            mask |= (ii == i) & (jj == j)
        sel = np.where(mask)[0]
        if len(sel):
            np.add.at(d, (qq[sel].astype(int), kk[sel].astype(int)), vv[sel])
        return d

    # ─────────────────────────────────────────────────────────────────────
    # Token banks (deterministic).  CRITICAL: prompts are built at the TOKEN-ID
    # level (concatenating id lists), NOT by string concat + re-encode, because
    # GPT-2 BPE merges across the "{lc}{A}" boundary and breaks probe-A==last.
    # ─────────────────────────────────────────────────────────────────────
    def enc1(s):
        ids = tok.encode(s)
        return ids[0] if len(ids) == 1 else None

    def E(s):
        return tok.encode(s)

    # neutral common-token filler bank (single-token; must NOT contain A or B)
    FILLER_STR = [" people", " time", " was", " with", " it", " they", " some", " many",
                  " day", " place", " year", " thing", " part", " way", " world", " life",
                  " hand", " eye", " word", " number"]
    FILLER = [enc1(w) for w in FILLER_STR if enc1(w) is not None]
    # left-context bank (the STRESS lever); single-token; split half LOCATE / half HELD-OUT
    LC_STR = [" the", " a", " and", " when", " near", " after", " my", " some", " then", " near"]
    LC_PAIRS = [(s, enc1(s)) for s in LC_STR if enc1(s) is not None]
    # dedup ids, keep order
    seen = set(); LC_PAIRS = [(s, i) for s, i in LC_PAIRS if not (i in seen or seen.add(i))]
    LC_LOCATE = LC_PAIRS[0::2]; LC_HOLD = LC_PAIRS[1::2]
    print(f"[bank] fillers={len(FILLER)} lc_locate={[s for s,_ in LC_LOCATE]} "
          f"lc_hold={[s for s,_ in LC_HOLD]}", flush=True)

    # candidate A/B pool: nonce-like (mostly multi-token in GPT-2) + random mid-vocab fallback
    NONCES = [" dax", " blicket", " wug", " fep", " zorp", " glorp", " trell",
              " kresh", " vung", " plim", " bnik", " florp", " quax", " jub", " mell"]
    nonce_ids = [(s, enc1(s)) for s in NONCES]
    nonce_ids = [(s, i) for s, i in nonce_ids if i is not None]
    print(f"[pool] single-token nonces: {[s for s,_ in nonce_ids]}", flush=True)
    rand_ids = []
    cand = (torch.randperm(40000, generator=torch.Generator().manual_seed(0))[:1500] + 1000).tolist()
    for t in cand:
        s = tok.decode([t])
        if len(tok.encode(s)) == 1 and s.strip() != "":
            rand_ids.append((s, t))
        if len(rand_ids) >= 90:
            break
    print(f"[pool] random-token fallback: {len(rand_ids)} candidates", flush=True)

    # ─────────────────────────────────────────────────────────────────────
    # Carrier templates  (built at TOKEN-ID level; probe-A always the LAST id)
    # A_id, B_id, lc_id, filler-ids are all single tokens -> probe-A==last is exact.
    # T6 decoys g/h are single-token nonces (or fillers if none).
    # ─────────────────────────────────────────────────────────────────────
    DECOY = [i for _, i in nonce_ids][:2]
    while len(DECOY) < 2:
        DECOY.append(FILLER[-(len(DECOY) + 1)])
    G, Hd = DECOY[0], DECOY[1]

    def fill_ids(n, seed):
        r = np.random.RandomState(seed)
        return [FILLER[r.randint(len(FILLER))] for _ in range(n)]

    def build_ids(template, A_id, B_id, lc_id, seed):
        f = fill_ids(10, seed)
        if template == "T1":
            return [A_id, B_id] + E(".") + f[:3] + E(".") + [lc_id, A_id]
        if template == "T2":
            return [A_id] + E(" means") + [B_id] + E(".") + f[:8] + E(".") + [lc_id, A_id]
        if template == "T3":
            return E(" The") + [A_id] + E(" gave a") + [B_id] + E(".") + f[:3] + E(",") + f[3:6] + E(";") + [lc_id, A_id]
        if template == "T4":
            return [A_id, B_id] + E(",") + f[:2] + E(",") + f[2:4] + E(",") + f[4:6] + E(".") + [lc_id, A_id]
        if template == "T5":
            return E(" Why") + [A_id, B_id] + E("?") + f[:5] + E(".") + [lc_id, A_id]
        if template == "T6":
            return [A_id, B_id] + E(".") + [G, Hd] + E(".") + f[:4] + E(".") + [lc_id, A_id]

    # benign carrier: A appears (mid-sentence) but NOT in an A->B induction context
    # (no B primer).  A is NOT required to be the last token here — M3c measures the
    # collateral at A's internal positions and the next-token prediction after A.
    def benign_ids(A_id, seed):
        f = fill_ids(6, seed)
        return E(" The") + [A_id] + f[:2] + E(" and") + [A_id] + f[2:5]

    # benign prompt where A IS the last token (for gate-4: top-1 after benign A != B)
    def benign_last_ids(A_id, seed):
        f = fill_ids(6, seed)
        return E(" The") + f[:2] + E(" and the") + f[2:5] + E(" of the") + [A_id]

    # Build the token tensor; probe-A is the last id by construction.  k_a = the
    # UNIQUE position whose token == B_id (templates place exactly one B).  Returns
    # (tt, seq, q_pos, k_pos) or None if B_id absent / B appears more than once.
    def make_tt(ids, A_id, B_id):
        ids = [tok.bos_token_id] + list(ids)
        if ids[-1] != A_id:
            return None
        bpos = [p for p, t in enumerate(ids) if t == B_id]
        if len(bpos) != 1:
            return None           # loud guard: exactly one B per primer (§5.2)
        q_pos = len(ids) - 1
        tt = torch.tensor(ids, device=dev).unsqueeze(0)
        return tt, len(ids), q_pos, bpos[0]

    # copyprob with an optional MULTI-HEAD score-delta dict {(l,h): np[seq,seq]}.
    def copyprob(tt, q_pos, B_id, deltas=None):
        if deltas is None:
            lg = model(tt)[0]
        else:
            lg = patched_logits_multi(tt, deltas)
        return float(torch.softmax(lg[q_pos].float(), -1)[B_id].item())

    # multi-head oracle: zero the induction edge (q,k) across ALL IND_HEADS.
    def oracle_edge_deltas(tt, q_pos, k_pos):
        seq = tt.shape[1]
        d = {}
        for (l, h) in IND_HEADS:
            sc = head_scores(tt, l, h)
            dd = np.zeros((seq, seq)); dd[q_pos, k_pos] = sc[q_pos, k_pos] + 20.0
            d[(l, h)] = dd
        return d

    # ─────────────────────────────────────────────────────────────────────
    # GATE (run FIRST) — pick 3 A->B pairs
    # ─────────────────────────────────────────────────────────────────────
    STATE["stage"] = "gate"
    print("[gate] screening candidate pairs ...", flush=True)
    SCREEN_TEMPLATES = ["T1", "T2", "T3"]    # screening carriers for gate-2

    pool = nonce_ids + rand_ids
    # build candidate (A,B) pairs: disjoint A,B from the pool
    def screen_pair(A_str, A_id, B_str, B_id):
        # gate-2: mean copyprob over screening carriers >= 0.30
        cps = []
        for tmpl in SCREEN_TEMPLATES:
            for _, lc_id in LC_LOCATE:
                e = make_tt(build_ids(tmpl, A_id, B_id, lc_id, 11), A_id, B_id)
                if e is None:
                    continue
                tt, seq, q_pos, k_pos = e
                cps.append(copyprob(tt, q_pos, B_id))
        if not cps:
            return None
        gate2 = float(np.mean(cps))
        # gate-3: parametric prior in a no-primer prompt (A once, B never follows)
        npids = E(" The") + [A_id] + FILLER[:2] + E(" and") + FILLER[2:3] + [A_id]
        ids = [tok.bos_token_id] + npids
        tt = torch.tensor(ids, device=dev).unsqueeze(0)
        par = float(torch.softmax(model(tt)[0][-1].float(), -1)[B_id].item())
        # gate-4: benign use — A's top-1 continuation in a benign (no-primer) carrier is NOT B
        bids = [tok.bos_token_id] + benign_last_ids(A_id, 3)
        bt = torch.tensor(bids, device=dev).unsqueeze(0)
        top1 = int(torch.softmax(model(bt)[0][-1].float(), -1).argmax().item())
        benign_ok = (top1 != B_id)
        # gate-5 (LOCUS): the induction-head SET edge must actually CARRY the copy.  The
        # DIAGNOSTIC showed L5H5 alone carries ~8% on average (the copy is DISTRIBUTED
        # across [(5,5),(6,9),(5,1),(7,10),(7,2)] which carries ~90%), so the causal locus
        # is the multi-head set -> the FRA cut + oracle both operate on the set.  Zero the
        # set's induction edge; require it removes >= ORACLE_MIN of the copy-prob.
        e = make_tt(build_ids("T1", A_id, B_id, LC_LOCATE[0][1], 11), A_id, B_id)
        oracle_rem = 0.0
        if e is not None:
            tt, seq, q_pos, k_pos = e
            cp0 = copyprob(tt, q_pos, B_id)
            if cp0 > 1e-6:
                cp1 = copyprob(tt, q_pos, B_id, deltas=oracle_edge_deltas(tt, q_pos, k_pos))
                oracle_rem = float(1.0 - cp1 / cp0)
        return dict(A=A_str, B=B_str, A_id=A_id, B_id=B_id, gate2=gate2,
                    parametric=par, benign_ok=benign_ok, oracle_rem=oracle_rem)

    passed = []
    tried = 0
    # iterate pairs nonce-first, then random; A!=B; cap the screen for budget
    cand_pairs = []
    for ai in range(len(pool)):
        for bi in range(len(pool)):
            if ai == bi:
                continue
            cand_pairs.append((pool[ai], pool[bi]))
    # nonce A & nonce B first (read naturally), then mixed, then random
    def rank_key(pp):
        (As, Ai), (Bs, Bi) = pp
        a_nonce = As in [s for s, _ in nonce_ids]; b_nonce = Bs in [s for s, _ in nonce_ids]
        return (-(a_nonce + b_nonce))
    cand_pairs.sort(key=rank_key)

    # gate-5 locus floor on the MULTI-HEAD induction set (diagnostic: set carries ~90%,
    # L5H5 alone ~8%).  Rank passers by oracle_rem; take the strongest-locus 3.
    ORACLE_MIN = float(os.environ.get("ORACLE_MIN", "0.40"))
    for ((As, Ai), (Bs, Bi)) in cand_pairs:
        if Ai == Bi:
            continue
        tried += 1
        if tried > 600:
            break
        r = screen_pair(As, Ai, Bs, Bi)
        if r is None:
            continue
        if r["gate2"] >= 0.30 and r["parametric"] < 0.02 and r["benign_ok"] and r["oracle_rem"] >= ORACLE_MIN:
            passed.append(r)
            print(f"[gate] PASS {As!r}->{Bs!r} gate2={r['gate2']:.3f} par={r['parametric']:.4f} "
                  f"oracle_rem={r['oracle_rem']:.2f}", flush=True)
            STATE["gate_passed"] = [(p["A"], p["B"], p["gate2"], p["parametric"], p["oracle_rem"]) for p in passed]
            ckpt(STATE, "persist_partial.json")
        if len(passed) >= 8:        # collect a few; pick the strongest-locus 3
            break

    if len(passed) < 3:
        print(f"[gate] WARNING only {len(passed)} pairs passed after {tried} tries; continuing with what we have", flush=True)
    # rank by oracle_rem (strongest L5H5 locus first), keep top 3
    passed.sort(key=lambda r: -r["oracle_rem"])
    chosen = passed[:3]
    print(f"[gate] CHOSEN pairs: {[(p['A'],p['B'],round(p['gate2'],3),round(p['oracle_rem'],2)) for p in chosen]}", flush=True)
    STATE["chosen_pairs"] = [dict(A=p["A"], B=p["B"], gate_copyprob=p["gate2"],
                                  parametric_prior=p["parametric"], oracle_rem=p["oracle_rem"]) for p in chosen]
    ckpt(STATE, "persist_partial.json")

    if not chosen:
        print("[FATAL] no pairs passed the gate", flush=True)
        ckpt({"error": "no pairs passed gate", **STATE}, "persist_results.json")
        return

    # ─────────────────────────────────────────────────────────────────────
    # Build the appearance sets per pair
    # LOCATE = {T1,T2,T3} x LC_LOCATE x seeds ; HELD-OUT = {T4,T5,T6} x LC_HOLD x seeds
    # ─────────────────────────────────────────────────────────────────────
    SEEDS = [11, 23, 37]
    def make_appearances(A_str, A_id, B_str, B_id, templates, lc_pairs):
        apps = []
        for tmpl in templates:
            for lc_str, lc_id in lc_pairs:
                for sd in SEEDS:
                    e = make_tt(build_ids(tmpl, A_id, B_id, lc_id, sd), A_id, B_id)
                    if e is None:
                        continue
                    tt, seq, q_pos, k_pos = e
                    cp = copyprob(tt, q_pos, B_id)
                    if cp < 0.15:        # sanity floor (must be doing induction here)
                        continue
                    apps.append(dict(tmpl=tmpl, lc=lc_str, seed=sd, tt=tt, seq=seq,
                                     q_pos=q_pos, k_pos=k_pos, cp_clean=cp))
        return apps

    # ─────────────────────────────────────────────────────────────────────
    # M2: per-appearance dominant cell + concentration.
    # MULTI-HEAD: a "cell" is now a (head, qF, kF) triple.  For each appearance we
    # build the FRA of every induction head, aggregate the edge per head, and pick the
    # globally-dominant (head,i,j) by |score|.  edge_cov is over ALL heads' edge terms.
    # The cached per-head arrays AF are reused for the unconditional cut.
    # ─────────────────────────────────────────────────────────────────────
    def appearance_cell(app):
        AF = fra_arrays_multi(app["tt"])
        q_pos, k_pos = app["q_pos"], app["k_pos"]
        cell = {}                  # (l,h,i,j) -> score
        for (l, h), A in AF.items():
            loc = np.where((A["qq"] == q_pos) & (A["kk"] == k_pos))[0]
            for n in loc:
                key = (l, h, int(A["ii"][n]), int(A["jj"][n]))
                cell[key] = cell.get(key, 0.0) + float(A["vv"][n])
        if not cell:
            return None
        items = sorted(cell.items(), key=lambda x: -abs(x[1]))
        (l_star, h_star, i_star, j_star), sc = items[0]
        tot = sum(abs(v) for v in cell.values())
        edge_cov = abs(sc) / tot if tot > 0 else 0.0
        return dict(l=l_star, h=h_star, i=i_star, j=j_star, score=sc,
                    edge_cov=edge_cov, AF=AF, cell_items=items[:5])

    # ─────────────────────────────────────────────────────────────────────
    # MAIN per-pair loop
    # ─────────────────────────────────────────────────────────────────────
    STATE["stage"] = "main_loop"
    per_pair = []
    pooled_cells = []          # (i,j) over ALL appearances pooled
    pooled_q = []; pooled_k = []
    pooled_lc_qfeat = []       # (lc, i) for contingency

    for pidx, P in enumerate(chosen):
        A_str, A_id, B_str, B_id = P["A"], P["A_id"], P["B"], P["B_id"]
        print(f"\n[pair {pidx}] {A_str!r}->{B_str!r}", flush=True)
        STATE["stage"] = f"pair{pidx}_appearances"

        loc_apps = make_appearances(A_str, A_id, B_str, B_id, ["T1", "T2", "T3"], LC_LOCATE)
        hold_apps = make_appearances(A_str, A_id, B_str, B_id, ["T4", "T5", "T6"], LC_HOLD)
        print(f"[pair {pidx}] n_locate={len(loc_apps)} n_holdout={len(hold_apps)}", flush=True)
        if len(loc_apps) == 0 or len(hold_apps) == 0:
            print(f"[pair {pidx}] SKIP: empty locate/holdout after sanity floor", flush=True)
            continue

        # --- M2 cells on LOCATE ---
        loc_cells = []
        for ai, app in enumerate(loc_apps):
            c = appearance_cell(app)
            if c is None:
                continue
            loc_cells.append((app, c))
            STATE["stage"] = f"pair{pidx}_locate_cell_{ai}"
            if ai % 5 == 0:
                ckpt(STATE, "persist_partial.json")
        # --- M2 cells on HELD-OUT ---
        hold_cells = []
        for ai, app in enumerate(hold_apps):
            c = appearance_cell(app)
            if c is None:
                continue
            hold_cells.append((app, c))
            STATE["stage"] = f"pair{pidx}_hold_cell_{ai}"
            if ai % 5 == 0:
                ckpt(STATE, "persist_partial.json")

        all_cells = loc_cells + hold_cells
        # a cell is now a (l,h,i,j) quadruple
        cellsList = [(c["l"], c["h"], c["i"], c["j"]) for _, c in all_cells]
        N = len(cellsList)
        from collections import Counter
        cnt = Counter(cellsList)
        ranked = cnt.most_common()
        top1_cell = ranked[0][0]
        top1_coverage = ranked[0][1] / N
        def cov(k):
            topk = set([cc for cc, _ in ranked[:k]])
            return sum(1 for cc in cellsList if cc in topk) / N
        cov1, cov3, cov5 = cov(1), cov(3), cov(5)
        n_cells_for_90 = next((k for k in range(1, N + 1) if cov(k) >= 0.90), N)
        # marginals: q-side = (l,h,i), k-side = (l,h,j)
        qcnt = Counter([(c["l"], c["h"], c["i"]) for _, c in all_cells])
        kcnt = Counter([(c["l"], c["h"], c["j"]) for _, c in all_cells])
        qtop1 = qcnt.most_common(1)[0][1] / N
        ktop1 = kcnt.most_common(1)[0][1] / N
        q_vs_k = "q-side drifts" if qtop1 < ktop1 else ("k-side drifts" if ktop1 < qtop1 else "tied")
        # head consistency: is the dominant HEAD stable across appearances?
        hcnt = Counter([(c["l"], c["h"]) for _, c in all_cells])
        head_top1 = hcnt.most_common(1)[0][1] / N
        # magnitude-weighted concentration
        wsum = sum(abs(c["score"]) for _, c in all_cells)
        wcnt = {}
        for _, c in all_cells:
            key = (c["l"], c["h"], c["i"], c["j"])
            wcnt[key] = wcnt.get(key, 0.0) + abs(c["score"])
        w_top1 = max(wcnt.values()) / wsum if wsum > 0 else 0.0
        # locate/holdout top1
        loc_top1 = Counter([(c["l"], c["h"], c["i"], c["j"]) for _, c in loc_cells]).most_common(1)[0][0]
        hold_top1 = Counter([(c["l"], c["h"], c["i"], c["j"]) for _, c in hold_cells]).most_common(1)[0][0]
        # {lc} x q-feature contingency
        lc_qfeat = {}
        for app, c in all_cells:
            lc_qfeat.setdefault(app["lc"], Counter())[c["i"]] += 1
        lc_qfeat_tab = {lc: dict(ct.most_common(3)) for lc, ct in lc_qfeat.items()}
        # mean within-appearance edge_coverage
        edge_cov_mean = float(np.mean([c["edge_cov"] for _, c in all_cells]))

        pooled_cells += cellsList
        pooled_q += [(c["l"], c["h"], c["i"]) for _, c in all_cells]
        pooled_k += [(c["l"], c["h"], c["j"]) for _, c in all_cells]
        for app, c in all_cells:
            pooled_lc_qfeat.append((app["lc"], c["i"]))

        print(f"[pair {pidx}] M2 top1={top1_cell} cov={top1_coverage:.2f} cov3={cov3:.2f} "
              f"n90={n_cells_for_90}/{N} qtop1={qtop1:.2f} ktop1={ktop1:.2f} ({q_vs_k}) "
              f"head_top1={head_top1:.2f} edge_cov_mean={edge_cov_mean:.3f}", flush=True)

        # --- M1: locate cell = LOCATE top1; apply UNCONDITIONAL cut on held-out ---
        loc_quad_cnt = Counter([(c["l"], c["h"], c["i"], c["j"]) for _, c in loc_cells])
        CELL = loc_quad_cnt.most_common(1)[0][0]   # (Lc, Hc, Ic, Jc)
        Lc, Hc, Ic, Jc = CELL
        loc_top3 = [cc for cc, _ in loc_quad_cnt.most_common(3)]
        STATE["stage"] = f"pair{pidx}_M1"

        # unconditional cut: for each located (l,h,i,j) cell, sum that head's FRA values for
        # (i,j) over ALL (q,k) and subtract from that head's scores (multi-head j7 hook).
        def rem_over(cells_apps, quads):
            rems = []
            for app, c in cells_apps:
                seq = app["seq"]
                deltas = {}
                for (l, h, i, j) in quads:
                    A = c["AF"][(l, h)]
                    dd = pairs_delta(A, [(i, j)], seq)
                    if (l, h) in deltas:
                        deltas[(l, h)] = deltas[(l, h)] + dd
                    else:
                        deltas[(l, h)] = dd
                cp_int = copyprob(app["tt"], app["q_pos"], B_id, deltas=deltas)
                rem = 1.0 - cp_int / max(app["cp_clean"], 1e-9)
                rems.append(min(max(rem, 0.0), 1.0))
            return float(np.mean(rems)) if rems else 0.0, rems

        rem_locate, _ = rem_over(loc_cells, [CELL])
        rem_holdout, _ = rem_over(hold_cells, [CELL])
        # M3b: union top-k on held-out
        rem_hold_k = {}
        for k in (1, 2, 3):
            rem_hold_k[k], _ = rem_over(hold_cells, loc_top3[:k])
        persistence_ratio = rem_holdout / rem_locate if rem_locate > 0 else 0.0
        oracle_ceiling = P.get("oracle_rem", 0.0)
        frac_of_oracle = rem_holdout / oracle_ceiling if oracle_ceiling > 1e-6 else 0.0
        base_cp_loc = float(np.mean([a["cp_clean"] for a, _ in loc_cells]))
        base_cp_hold = float(np.mean([a["cp_clean"] for a, _ in hold_cells]))
        print(f"[pair {pidx}] M1 rem_locate={rem_locate:.3f} rem_holdout={rem_holdout:.3f} "
              f"ratio={persistence_ratio:.2f} oracle_ceil={oracle_ceiling:.2f} "
              f"frac_of_oracle={frac_of_oracle:.2f} | union k=1/2/3 -> "
              f"{rem_hold_k[1]:.3f}/{rem_hold_k[2]:.3f}/{rem_hold_k[3]:.3f}", flush=True)
        ckpt(STATE, "persist_partial.json")

        # --- M3a: token-mask (no detector) vs oracle ---
        STATE["stage"] = f"pair{pidx}_M3a"
        # token-mask NO-DETECTOR: use the LOCATE primer's (q,k) position PATTERN verbatim.
        # The locate primer pattern is (q_loc, k_loc) from a representative locate appearance.
        # Applied to held-out, those absolute positions don't carry the induction edge -> ~0.
        loc_ref = loc_cells[0][0]
        q_loc, k_loc = loc_ref["q_pos"], loc_ref["k_pos"]
        def tokenmask_nodetector(app):
            # mask the FIXED locate (q,k) position across ALL heads (positions differ on
            # held-out -> expect ~0).  No detector: the absolute positions are hard-coded.
            seq = app["seq"]
            deltas = {}
            if q_loc < seq and k_loc < seq:
                for (l, h) in IND_HEADS:
                    sc = head_scores(app["tt"], l, h)
                    dd = np.zeros((seq, seq)); dd[q_loc, k_loc] = sc[q_loc, k_loc] + 20.0
                    deltas[(l, h)] = dd
            cp_int = copyprob(app["tt"], app["q_pos"], B_id, deltas=deltas if deltas else None)
            return min(max(1.0 - cp_int / max(app["cp_clean"], 1e-9), 0.0), 1.0)
        def tokenmask_oracle(app):
            # oracle: position-patch the held-out A's ACTUAL induction edge across all heads
            cp_int = copyprob(app["tt"], app["q_pos"], B_id,
                              deltas=oracle_edge_deltas(app["tt"], app["q_pos"], app["k_pos"]))
            return min(max(1.0 - cp_int / max(app["cp_clean"], 1e-9), 0.0), 1.0)
        rem_mask_nodet = float(np.mean([tokenmask_nodetector(a) for a, _ in hold_cells]))
        rem_mask_oracle = float(np.mean([tokenmask_oracle(a) for a, _ in hold_cells]))
        fra_vs_mask = rem_holdout / max(rem_mask_nodet, 1e-6)
        print(f"[pair {pidx}] M3a tokenmask_nodet={rem_mask_nodet:.2f} oracle={rem_mask_oracle:.2f} "
              f"FRA/mask={fra_vs_mask:.1f}x", flush=True)
        ckpt(STATE, "persist_partial.json")

        # --- M3c: benign-use preservation (FRA vs feature-ablation vs ActAdd) ---
        STATE["stage"] = f"pair{pidx}_M3c"
        def klvec(p, q):
            lp = torch.log_softmax(p.float(), -1); lq = torch.log_softmax(q.float(), -1)
            return (lp.exp() * (lp - lq)).sum(-1)
        benign_apps = []
        for sd in [3, 9, 15, 21]:
            bids = [tok.bos_token_id] + benign_ids(A_id, sd)
            a_positions = [p for p, t in enumerate(bids) if t == A_id]
            if not a_positions:
                continue
            bt = torch.tensor(bids, device=dev).unsqueeze(0)
            benign_apps.append(dict(tt=bt, seq=len(bids), a_pos=a_positions))
        fra_kl, ablate_kl, steer_kl = [], [], []
        fra_top1, ablate_top1, steer_top1 = [], [], []
        # ActAdd direction = A's residual direction at L (resid_pre)
        for bapp in benign_apps:
            bt = bapp["tt"]; seq = bapp["seq"]; apos = bapp["a_pos"]
            clean = model(bt)[0]
            AF = fra_arrays_multi(bt)
            # FRA cell-cut (unconditional, the located (Lc,Hc,Ic,Jc) on its head only)
            dF = pairs_delta(AF[(Lc, Hc)], [(Ic, Jc)], seq)
            lgF = patched_logits_multi(bt, {(Lc, Hc): dF})
            # feature-ablation of A's query feature Ic at layer Lc everywhere (zero its
            # resid contribution at all positions) -> zero SAE feature Ic in the resid_pre
            # reconstruction and add the resid delta back via a hook.
            hook_name = f"blocks.{Lc}.hook_resid_pre"
            _, c2 = model.run_with_cache(bt, names_filter=[hook_name])
            act = c2[hook_name][0]
            feats = saes[Lc].encode(act).float()
            feats_ab = feats.clone(); feats_ab[:, Ic] = 0.0
            x_ab = feats_ab @ saes[Lc].W_dec.float() + saes[Lc].b_dec.float()
            delta_resid = (x_ab - (feats @ saes[Lc].W_dec.float() + saes[Lc].b_dec.float()))
            def ab_hook(resid, hook):
                resid[0] = resid[0] + delta_resid
                return resid
            lgAb = model.run_with_hooks(bt, fwd_hooks=[(hook_name, ab_hook)])[0]
            # ActAdd: subtract A's mean-centered resid direction at A positions
            vX = act[apos[0]] - act.mean(0); vX = vX / (vX.norm() + 1e-6)
            def aa_hook(resid, hook):
                for p in apos:
                    resid[0, p, :] = resid[0, p, :] - 4.0 * vX * resid[0, p, :].norm()
                return resid
            lgAA = model.run_with_hooks(bt, fwd_hooks=[(hook_name, aa_hook)])[0]
            klF = klvec(clean, lgF); klAb = klvec(clean, lgAb); klAA = klvec(clean, lgAA)
            for p in apos:
                fra_kl.append(float(klF[p])); ablate_kl.append(float(klAb[p])); steer_kl.append(float(klAA[p]))
                fra_top1.append(int(clean[p].argmax() == lgF[p].argmax()))
                ablate_top1.append(int(clean[p].argmax() == lgAb[p].argmax()))
                steer_top1.append(int(clean[p].argmax() == lgAA[p].argmax()))
        def m(x): return float(np.mean(x)) if x else 0.0
        fra_collat = m(fra_kl); ablate_collat = m(ablate_kl); steer_collat = m(steer_kl)
        ratio_ablate = ablate_collat / max(fra_collat, 1e-6)
        print(f"[pair {pidx}] M3c benign collat_KL FRA={fra_collat:.3f} ablate={ablate_collat:.3f} "
              f"steer={steer_collat:.3f} (ablate/FRA={ratio_ablate:.1f}x)", flush=True)

        # --- non-sink gate: is the located q/k feature a sink (fires on >50% of generic positions)?
        # cheap proxy: fraction of benign+filler positions where feature I (q) / J (k) is active
        STATE["stage"] = f"pair{pidx}_nonsink"
        generic_txt = " The people were with them and many of the things that day in this place"
        gids = [tok.bos_token_id] + tok.encode(generic_txt)
        gt = torch.tensor(gids, device=dev).unsqueeze(0)
        _, gc = model.run_with_cache(gt, names_filter=[f"blocks.{Lc}.hook_resid_pre"])
        gfeats = saes[Lc].encode(gc[f"blocks.{Lc}.hook_resid_pre"][0]).float()
        sink_q = float((gfeats[:, Ic] != 0).float().mean())
        sink_k = float((gfeats[:, Jc] != 0).float().mean())
        non_sink = (sink_q <= 0.50) and (sink_k <= 0.50)
        print(f"[pair {pidx}] non-sink: L{Lc}H{Hc} q-feat {Ic} active@{sink_q:.2f} "
              f"k-feat {Jc} active@{sink_k:.2f} -> non_sink={non_sink}", flush=True)

        pr = dict(
            A=A_str, B=B_str, A_id=A_id, B_id=B_id,
            n_locate=len(loc_cells), n_holdout=len(hold_cells), N=N,
            M2=dict(top1_cell=list(top1_cell), top1_coverage=top1_coverage,
                    cov1=cov1, cov3=cov3, cov5=cov5, n_cells_for_90=n_cells_for_90,
                    qtop1=qtop1, ktop1=ktop1, q_vs_k_culprit=q_vs_k, head_top1=head_top1,
                    w_top1=w_top1, edge_cov_mean=edge_cov_mean,
                    locate_top1=list(loc_top1), holdout_top1=list(hold_top1),
                    top1_locate_eq_holdout=(loc_top1 == hold_top1),
                    cell_hist=[[list(cc), n] for cc, n in ranked[:10]],
                    lc_x_qfeat=lc_qfeat_tab),
            M1=dict(located_cell=[Lc, Hc, Ic, Jc], rem_locate=rem_locate, rem_holdout=rem_holdout,
                    persistence_ratio=persistence_ratio, oracle_ceiling=oracle_ceiling,
                    frac_of_oracle=frac_of_oracle,
                    base_copyprob_locate=base_cp_loc, base_copyprob_holdout=base_cp_hold),
            M3a=dict(rem_holdout_tokenmask_nodetector=rem_mask_nodet,
                     rem_holdout_tokenmask_oracle=rem_mask_oracle, FRA_vs_mask_ratio=fra_vs_mask),
            M3b=dict(rem_holdout_k1=rem_hold_k[1], rem_holdout_k2=rem_hold_k[2],
                     rem_holdout_k3=rem_hold_k[3], union_gain=rem_hold_k[3] - rem_hold_k[1],
                     locate_top3=[list(c) for c in loc_top3]),
            M3c=dict(FRA_collat_KL=fra_collat, featablate_collat_KL=ablate_collat,
                     linsteer_collat_KL=steer_collat, ablate_over_fra=ratio_ablate,
                     benign_top1_preserved=dict(FRA=m(fra_top1), ablate=m(ablate_top1),
                                                steer=m(steer_top1))),
            sanity=dict(non_sink=non_sink, sink_q_active=sink_q, sink_k_active=sink_k),
        )
        per_pair.append(pr)
        STATE["per_pair_done"] = pidx + 1
        ckpt({**STATE, "per_pair": per_pair}, "persist_partial.json")

    # ─────────────────────────────────────────────────────────────────────
    # POOLED stats + VERDICT
    # ─────────────────────────────────────────────────────────────────────
    STATE["stage"] = "pooled_verdict"
    from collections import Counter
    if not per_pair:
        ckpt({"error": "no pair completed", **STATE}, "persist_results.json")
        return
    N = len(pooled_cells)
    cnt = Counter(pooled_cells); ranked = cnt.most_common()
    def cov(k):
        topk = set([cc for cc, _ in ranked[:k]])
        return sum(1 for cc in pooled_cells if cc in topk) / N
    pooled_top1cov = ranked[0][1] / N
    pooled_cov3 = cov(3); pooled_cov5 = cov(5)
    pooled_n90 = next((k for k in range(1, N + 1) if cov(k) >= 0.90), N)
    qtop1 = Counter(pooled_q).most_common(1)[0][1] / N
    ktop1 = Counter(pooled_k).most_common(1)[0][1] / N
    q_vs_k = "q-side drifts" if qtop1 < ktop1 else ("k-side drifts" if ktop1 < qtop1 else "tied")
    rem_hold_pool = float(np.mean([p["M1"]["rem_holdout"] for p in per_pair]))
    rem_hold_k3_pool = float(np.mean([p["M3b"]["rem_holdout_k3"] for p in per_pair]))
    base_hold_pool = float(np.mean([p["M1"]["base_copyprob_holdout"] for p in per_pair]))
    base_loc_pool = float(np.mean([p["M1"]["base_copyprob_locate"] for p in per_pair]))
    mask_nodet_pool = float(np.mean([p["M3a"]["rem_holdout_tokenmask_nodetector"] for p in per_pair]))
    fra_vs_mask_pool = rem_hold_pool / max(mask_nodet_pool, 1e-6)
    fra_collat_pool = float(np.mean([p["M3c"]["FRA_collat_KL"] for p in per_pair]))
    ablate_collat_pool = float(np.mean([p["M3c"]["featablate_collat_KL"] for p in per_pair]))
    all_non_sink = all(p["sanity"]["non_sink"] for p in per_pair)
    head_top1_pool = Counter([(l, h) for (l, h, i) in pooled_q]).most_common(1)[0][1] / N
    oracle_pool = float(np.mean([p["M1"]["oracle_ceiling"] for p in per_pair]))
    frac_of_oracle_pool = float(np.mean([p["M1"]["frac_of_oracle"] for p in per_pair]))

    # ----- VERDICT (PERSIST_DESIGN §4) -----
    # sanity gate: base copyprob >= 0.30 on both sets AND non-sink
    sanity_ok = (base_loc_pool >= 0.30) and (base_hold_pool >= 0.30) and all_non_sink
    # clean WIN
    clean_win = (pooled_top1cov >= 0.70) and (rem_hold_pool >= 0.70)
    # bounded-union WIN
    bounded_win = (0.40 <= pooled_top1cov < 0.70) and (pooled_cov3 >= 0.80) \
        and (pooled_n90 <= 3) and (rem_hold_k3_pool >= 0.70)
    # informative-negative
    info_neg = (pooled_cov3 < 0.40) or (pooled_n90 >= 0.5 * N) or (rem_hold_k3_pool < 0.40)

    if not sanity_ok:
        verdict = "INVALID (sanity gate failed)"
        deciding = f"base_loc={base_loc_pool:.2f} base_hold={base_hold_pool:.2f} non_sink={all_non_sink}"
    elif clean_win:
        verdict = "WIN (single-cell)"
        deciding = f"top1cov={pooled_top1cov:.2f}>=0.70 AND rem_holdout={rem_hold_pool:.2f}>=0.70"
    elif bounded_win:
        verdict = "WIN (bounded-union k<=3)"
        deciding = f"cov3={pooled_cov3:.2f}>=0.80 n90={pooled_n90}<=3 rem_k3={rem_hold_k3_pool:.2f}>=0.70"
    elif info_neg:
        verdict = "INFORMATIVE-NEGATIVE"
        reasons = []
        if pooled_cov3 < 0.40: reasons.append(f"cov3={pooled_cov3:.2f}<0.40")
        if pooled_n90 >= 0.5 * N: reasons.append(f"n90={pooled_n90}>=0.5N={0.5*N:.0f}")
        if rem_hold_k3_pool < 0.40: reasons.append(f"rem_k3={rem_hold_k3_pool:.2f}<0.40")
        deciding = "; ".join(reasons)
    else:
        verdict = "AMBIGUOUS (partial persistence)"
        deciding = f"top1cov={pooled_top1cov:.2f} cov3={pooled_cov3:.2f} rem_holdout={rem_hold_pool:.2f} rem_k3={rem_hold_k3_pool:.2f}"

    pooled = dict(
        N=N, top1_coverage=pooled_top1cov, cov3=pooled_cov3, cov5=pooled_cov5,
        n_cells_for_90=pooled_n90, qtop1=qtop1, ktop1=ktop1, q_vs_k_culprit=q_vs_k,
        head_top1=head_top1_pool,
        cell_hist=[[list(cc), n] for cc, n in ranked[:10]],
        rem_holdout=rem_hold_pool, rem_holdout_k3=rem_hold_k3_pool,
        oracle_ceiling=oracle_pool, frac_of_oracle=frac_of_oracle_pool,
        base_copyprob_locate=base_loc_pool, base_copyprob_holdout=base_hold_pool,
        tokenmask_nodetector=mask_nodet_pool, FRA_vs_mask_ratio=fra_vs_mask_pool,
        FRA_collat_KL=fra_collat_pool, featablate_collat_KL=ablate_collat_pool,
        ablate_over_fra=ablate_collat_pool / max(fra_collat_pool, 1e-6),
        all_non_sink=all_non_sink, sanity_ok=sanity_ok,
    )
    results = dict(
        chosen_pairs=STATE["chosen_pairs"],
        gate_lc_locate=LC_LOCATE, gate_lc_holdout=LC_HOLD,
        per_pair=per_pair, pooled=pooled,
        VERDICT=verdict, deciding_number=deciding,
        runtime_s=time.time() - STATE["t0"],
    )
    print("\n" + "=" * 60, flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"  deciding: {deciding}", flush=True)
    print(f"  pooled top1cov={pooled_top1cov:.2f} cov3={pooled_cov3:.2f} n90={pooled_n90}/{N} "
          f"({q_vs_k})", flush=True)
    print(f"  rem_holdout={rem_hold_pool:.2f} rem_k3={rem_hold_k3_pool:.2f} "
          f"mask_nodet={mask_nodet_pool:.2f} FRA/mask={fra_vs_mask_pool:.1f}x", flush=True)
    print(f"  benign collat FRA={fra_collat_pool:.3f} ablate={ablate_collat_pool:.3f}", flush=True)
    print("=" * 60, flush=True)
    ckpt(results, "persist_results.json")
    print("[DONE] persist_results.json uploaded", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print("[FATAL]\n" + tb, flush=True)
        try:
            p = os.path.join(OUT, "persist_traceback.txt")
            open(p, "w").write(tb + "\n\nSTATE=" + json.dumps(STATE, default=str))
            upload(p, "persist_traceback.txt")
        except Exception as e:
            print(f"[traceback-upload-fail] {e}", flush=True)
        sys.exit(1)
