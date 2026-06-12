"""IN-CONTEXT VARIABLE BINDING (A1) — the §3.3 SELECTIVITY WIN-TEST (SCREEN2 §3.3 / PREDICTOR2 §4).
The decisive head-to-head injection failed at 1.6x, now on the organism the pre-check cleared STRONG-GO
(R_gen 0.90, sibling-bleed 0.0009, 29/29 content-addressed).

QUESTION (judge-free, ground-truth next-token exact-match): at MATCHED target-suppression (drive the TARGET
binding's correct-entity retrieval down by a common amount), which method pays the least SIBLING collateral —
the FRA (target-entity x target-value) score-cell edit, the BEST-TUNED linear DoM "retrieve-bound-value"
steer, or whole-head ablation? The HARD control is the INTRA-INSTANCE siblings: the OTHER 3 bindings in the
SAME prompt (share everything but the conjunction — the hardest possible selectivity stressor).

Re-points the v1 selectivity harness (../fra_organisms/jobs/injection_selectivity.py) + reuses the pre-check's
render/cut machinery VERBATIM. Model gemma-2-2b-it + gemma-scope-2b-pt-res-canonical. Completion format
("... Who has the {Vq}? Answer:" -> next token = bound entity). Heads from the pre-check: L22H4 (+L18H6).

THREE interventions, each swept and read at a COMMON target-suppression t*:
 (a) FRA cell-cut: FRA-decompose the (answer-query x target-entity) AND (answer-query x target-value) edges
     -> top-M (qfeat x kfeat) pairs -> per-head [seq,seq] score-delta -> subtract c*delta at hook_attn_scores
     (the surgical multi-cell edit, NOT a naive single cut). Sweep c. Ceiling = the hard oracle cut (c=inf).
 (b) BEST-TUNED LINEAR DoM: v_L = mean(resid | target-binding answer-pos) - mean(resid | SAME prompt with the
     target binding's value swapped so the answer changes) = the "retrieve-THIS-bound-value" direction. Built
     on a TRAIN split disjoint from eval. Project-subtract alpha*vhat at all positions of layer L. Sweep
     layers {6,9,12} x alpha. BEST-TUNED = the (layer,alpha) with the lowest sibling-collateral at t*.
 (c) HEAD-ABLATION: zero the whole L22H4 head's hook_z output (and L22H4+L18H6) — the circuit-tracing baseline
     (also the A3 cell-cut-vs-head-ablation comparison). Full ablation -> single operating point per head-set.

MATCHED POINT: per method, trace (target-suppression, sibling-collateral). t* = a suppression level FRA
REACHES (the v1 reachability lesson). Interpolate each method's sibling-collateral at t*. HEADLINE =
collateral(best-linear)/collateral(FRA) on the intra-instance sibling control.
WIN iff FRA reaches t*, sibling-collateral ratio (linear/FRA) >= 2x AND >= 0.15 absolute advantage AND FRA <
best-linear AND FRA < head-ablation. NULL (reported honestly) iff < 2x.

target-suppression = 1 - mean(target P(correct) after / before).  sibling-collateral = 1 - mean(sibling
P(correct) after / before)  (the OTHER bindings in the SAME prompt, re-queried, under the SAME target edit).
capability = mean(sibling baseline-correct rate preserved) + degenerate-answer guard.

Resume-proof: early flush + ckpt BEFORE heavy loops; ckpt INSIDE every sweep-point / per-instance loop;
top-of-script try/except uploads FATAL.json (the v1 restart-loop fix).
"""
import os, sys, json, time, traceback
sys.path.insert(0, "/workspace/code")

OUT = os.environ.get("OUTDIR", ".")
CKPT = os.path.join(OUT, "binding_selectivity.json")

def log(*a):
    print(*a, flush=True)

log(f"[{time.strftime('%H:%M:%S')}] PYJOB START — binding_selectivity entered (OUT={OUT})")
try:
    os.makedirs(OUT, exist_ok=True)
    json.dump({"heartbeat": "entered", "t": time.strftime("%H:%M:%S")},
              open(os.path.join(OUT, "heartbeat.json"), "w"))
except Exception:
    pass

try:
    import torch, numpy as np
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    from fra.sae_lens_wrapper import GemmaScopeSAE
    from fra.core.fra import _build_fra_result

    # ---- config ---------------------------------------------------------
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-2-2b-it")
    N_INST     = int(os.environ.get("N_INST", "200"))    # candidate instances
    N_EVAL     = int(os.environ.get("N_EVAL", "60"))     # eval instances for the curves (cheap on L4)
    N_TRAIN    = int(os.environ.get("N_TRAIN", "40"))    # DISJOINT train instances for the DoM vector
    N_BIND     = 4
    SEED       = int(os.environ.get("SEED", "0"))
    FRA_HEADS  = [(22, 4), (18, 6)]                      # from the pre-check head-find (L22H4 dominant)
    ABL_HEADS_A = [(22, 4)]                              # head-ablation variant A
    ABL_HEADS_B = [(22, 4), (18, 6)]                     # head-ablation variant B
    M_PAIRS    = int(os.environ.get("M_PAIRS", "24"))   # top SAE feature-pairs per (head, key) cell
    FRA_CS     = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # FRA edit scale sweep
    LIN_LAYERS = [6, 9, 12]
    LIN_ALPHA  = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    T_STAR     = float(os.environ.get("T_STAR", "0.80")) # matched target-suppression (FRA reaches ~0.90 oracle)

    def ckpt():
        json.dump(state, open(CKPT, "w"), indent=2, default=float)

    log(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=dev, dtype=torch.float16)
    model.eval()
    tok = model.tokenizer
    NL = model.cfg.n_layers; NH = model.cfg.n_heads
    log(f"  loaded: {NL} layers x {NH} heads")

    # ---- vocab (single-token, runtime-verified) — identical to the pre-check ----
    NAME_POOL = ["Ann", "Joe", "Pete", "Tim", "Sam", "Tom", "Kate", "Mark", "Paul", "Jane",
                 "Bob", "Sue", "Dan", "Lucy", "Mike", "Anna", "Jack", "Mary", "Ben", "Rose",
                 "Carl", "Nina", "Eric", "Lisa", "Gary", "Emma", "Fred", "Ruth", "Adam", "Beth"]
    VALUE_POOL = ["ale", "pie", "jam", "tea", "ham", "fig", "cod", "oat", "bun", "egg",
                  "nut", "yam", "cake", "wine", "fish", "rice", "soup", "milk", "corn", "plum",
                  "pear", "lime", "kale", "beef", "duck", "lamb", "crab", "rye", "oil", "bread"]
    def single_tok(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        return (len(ids) == 1), (ids[0] if ids else None)
    names, name_tid = [], {}
    for w in NAME_POOL:
        ok, t = single_tok(w)
        if ok: names.append(w); name_tid[w] = t
    values, value_tid = [], {}
    for w in VALUE_POOL:
        ok, t = single_tok(w)
        if ok: values.append(w); value_tid[w] = t
    log(f"  single-token pool: {len(names)} names, {len(values)} values")

    rng = np.random.RandomState(SEED)
    def build_instance():
        E = list(rng.choice(names, N_BIND, replace=False))
        V = list(rng.choice(values, N_BIND, replace=False))
        qi = int(rng.randint(N_BIND))
        return dict(E=E, V=V, qi=qi)
    def render(E, V, qi):
        parts = [f"{E[i]} has the {V[i]}" for i in range(N_BIND)]
        body = ", ".join(parts) + "."
        return f"{body} Who has the {V[qi]}? Answer:", E[qi]
    def build_tokens(text):
        ids = tok.encode(text, add_special_tokens=True)
        return ids, torch.tensor(ids, device=dev).unsqueeze(0)
    def p_at(tt, tid, qpos=None):
        logits = model(tt)[0]
        q = (tt.shape[1] - 1) if qpos is None else qpos
        return torch.softmax(logits[q].float(), -1)[tid].item()
    def argmax_at(tt, qpos=None):
        logits = model(tt)[0]
        q = (tt.shape[1] - 1) if qpos is None else qpos
        return int(logits[q].argmax().item())
    def find_word_pos(ids, word, q, occurrence=0):
        wt = tok.encode(" " + word, add_special_tokens=False)
        if len(wt) != 1: return None
        t = wt[0]; hits = [i for i in range(q) if ids[i] == t]
        return hits[occurrence] if occurrence < len(hits) else None

    # ====================================================================
    # resume state
    # ====================================================================
    state = {"config": {"model": MODEL_NAME, "fra_heads": FRA_HEADS, "abl_a": ABL_HEADS_A, "abl_b": ABL_HEADS_B,
                        "fra_cs": FRA_CS, "lin_layers": LIN_LAYERS, "lin_alpha": LIN_ALPHA,
                        "m_pairs": M_PAIRS, "t_star": T_STAR, "n_eval": N_EVAL, "n_train": N_TRAIN,
                        "seed": SEED, "sae": "gemma-scope-2b-pt-res-canonical"},
             "instances": [], "eval_idx": None, "train_idx": None,
             "fra_curve": [], "fra_oracle": None,
             "dom_vecs": None, "lin_curve": [], "headabl": [], "cap": None}
    if os.path.exists(CKPT):
        try:
            state = json.load(open(CKPT))
            log(f"  resumed: fra={len(state['fra_curve'])} lin={len(state['lin_curve'])} headabl={len(state['headabl'])}")
        except Exception:
            pass
    ckpt()

    # ====================================================================
    # STAGE 1: build instances, baseline-filter, split eval/train (DISJOINT)
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 1 build + baseline-correct filter + eval/train split")
    if not state["instances"]:
        insts = [build_instance() for _ in range(N_INST)]
        seen = set(); uniq = []
        for it in insts:
            k = (tuple(it["E"]), tuple(it["V"]), it["qi"])
            if k not in seen: seen.add(k); uniq.append(it)
        # baseline-filter: keep only baseline-correct (argmax next token == answer)
        kept = []
        for ix, it in enumerate(uniq):
            prompt, ans = render(it["E"], it["V"], it["qi"])
            ids, tt = build_tokens(prompt)
            if tt.shape[1] > 200: continue
            am = argmax_at(tt)
            if am == name_tid[ans]:
                it = dict(it); it["base_p"] = float(p_at(tt, name_tid[ans]))
                kept.append(it)
            if len(kept) >= (N_EVAL + N_TRAIN + 5): break
        state["instances"] = kept
        ckpt()
        log(f"  baseline-correct instances kept: {len(kept)}")
    kept = state["instances"]
    if len(kept) < (N_EVAL + 10):
        # shrink eval if needed
        N_EVAL = max(20, len(kept) - N_TRAIN - 2)
    if state["eval_idx"] is None:
        idxs = list(range(len(kept)))
        state["train_idx"] = idxs[:N_TRAIN]
        state["eval_idx"] = idxs[N_TRAIN:N_TRAIN + N_EVAL]
        ckpt()
    EVAL = state["eval_idx"]; TRAIN = state["train_idx"]
    log(f"  eval n={len(EVAL)}  train n={len(TRAIN)} (disjoint)")

    # ---- helper: per-instance keys + tokens (target entity/value positions) ----
    def inst_prompt(ix):
        it = kept[ix]; E, V, qi = it["E"], it["V"], it["qi"]
        ans = E[qi]; ans_tid = name_tid[ans]
        prompt, _ = render(E, V, qi)
        ids, tt = build_tokens(prompt)
        qpos = tt.shape[1] - 1
        kE = find_word_pos(ids, E[qi], qpos, 0)
        kV = find_word_pos(ids, V[qi], qpos, 0)
        return it, E, V, qi, ans_tid, ids, tt, qpos, kE, kV

    # ---- siblings: re-query the OTHER bindings in the SAME prompt ----
    def sibling_prompts(ix):
        """list of (sib_tid, ids, tt, qpos, kE_target, kV_target) — same context, query each OTHER binding's
        value; locate the TARGET entity/value keys in this rendering (the cut is on the TARGET cell)."""
        it = kept[ix]; E, V, qi = it["E"], it["V"], it["qi"]
        out = []
        for j in range(N_BIND):
            if j == qi: continue
            prompt, _ = render(E, V, j)
            ids, tt = build_tokens(prompt)
            qpos = tt.shape[1] - 1
            sib_tid = name_tid[E[j]]
            kE = find_word_pos(ids, E[qi], qpos, 0)   # TARGET entity key (cut the target conjunction)
            kV = find_word_pos(ids, V[qi], qpos, 0)
            base = p_at(tt, sib_tid)
            out.append(dict(j=j, sib_tid=sib_tid, ids=ids, tt=tt, qpos=qpos, kE=kE, kV=kV, base=base))
        return out

    # ====================================================================
    # SCORE-CELL cut hooks (hard oracle cut: zero target entity+value cols at answer row) — pre-check verbatim
    # ====================================================================
    def hard_cut_hooks(heads, qpos, kE, kV):
        byL = {}
        for L, H in heads: byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    seq = s.shape[2]; qq = qpos if qpos < seq else seq - 1
                    for H in Hs:
                        if kE is not None and kE < seq: s[0, H, qq, kE] = -1e4
                        if kV is not None and kV < seq: s[0, H, qq, kV] = -1e4
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return hooks

    # ====================================================================
    # FRA machinery (= injection/pre-check fra_edge), build per-(head) score-delta over target cells
    # ====================================================================
    _SAE = {}
    def get_sae(L):
        if L not in _SAE:
            sl = L - 1
            try:
                _SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",
                                        f"layer_{sl}/width_16k/canonical", device=dev, normalize_activations=True)
            except Exception:
                _SAE[L] = GemmaScopeSAE("gemma-scope-2b-pt-res",
                                        f"layer_{sl}/width_16k/average_l0_68", device=dev, normalize_activations=True)
            log(f"    SAE layer {sl} loaded")
        return _SAE[L]
    def encode_layer(L, tt):
        hk = f"blocks.{L}.hook_resid_pre"
        a = model.run_with_cache(tt, names_filter=[hk])[1][hk][0].float()
        sae = get_sae(L); f = sae.encode(a).float()
        if sae._norm_coeff is not None: f = f / sae._norm_coeff
        return f, a
    def fra_edge(L, H, feats, Wdec, xh, tt):
        r = _build_fra_result(model, L, H, feats, Wdec, dev, top_k=None,
                              rms_activations=xh, dec_norms=None, chunk_size=8, verbose=False)
        f = r["fra_tensor_sparse"].coalesce(); idx = f.indices().cpu().numpy()
        return dict(qq=idx[0], kk=idx[1], ii=idx[2], jj=idx[3], vv=f.values().cpu().numpy())

    def build_fra_delta(tt, qpos, kE, kV):
        """per-head [seq,seq] score-delta of the top-M (qfeat x kfeat) pairs supporting the (qpos x kE) AND
        (qpos x kV) cells — the TARGET conjunction. Returns byL->{H:delta} (the surgical multi-cell edit)."""
        seq = tt.shape[1]
        layers = sorted(set(L for L, H in FRA_HEADS)); fcache = {}
        for L in layers:
            f, _ = encode_layer(L, tt)
            xh = f @ get_sae(L).W_dec.float() + get_sae(L).b_dec.float()
            fcache[L] = (f, xh)
        byL = {}
        for (L, H) in FRA_HEADS:
            f, xh = fcache[L]
            d = fra_edge(L, H, f, get_sae(L).W_dec.float(), xh, tt)
            P = set()
            for kpos in (kE, kV):
                if kpos is None: continue
                loc = np.where((d["qq"] == qpos) & (d["kk"] == kpos))[0]
                loc = loc[np.argsort(-np.abs(d["vv"][loc]))[:M_PAIRS]]
                for o in loc: P.add((int(d["ii"][o]), int(d["jj"][o])))
            delta = np.zeros((seq, seq))
            for n in range(len(d["vv"])):
                if (int(d["ii"][n]), int(d["jj"][n])) in P and d["kk"][n] in (kE, kV):
                    delta[d["qq"][n], d["kk"][n]] += d["vv"][n]
            byL.setdefault(L, {})[H] = delta
        return byL

    def fra_hooks(byL, c):
        hooks = []
        for L, hd in byL.items():
            td = {H: torch.tensor(dd, device=dev, dtype=torch.float32) * c for H, dd in hd.items()}
            def mk(td):
                def hook(s, hook):
                    sq = s.shape[2]
                    for H, sd in td.items():
                        n = min(sq, sd.shape[0])
                        s[0, H, :n, :n] = s[0, H, :n, :n] - sd[:n, :n].to(s.dtype)
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(td)))
        return hooks

    # ====================================================================
    # linear DoM "retrieve-THIS-bound-value" hooks
    # ====================================================================
    def dom_hooks(L, vhat, alpha):
        def hook(act, hook):
            proj = (act[0] @ vhat).unsqueeze(-1) * vhat
            act[0] = act[0] - alpha * proj
            return act
        return [(f"blocks.{L}.hook_resid_pre", hook)]

    # ====================================================================
    # head-ablation hooks: zero the head's hook_z output (the circuit-tracing baseline)
    # ====================================================================
    def headabl_hooks(heads):
        byL = {}
        for L, H in heads: byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(z, hook):
                    for H in Hs: z[0, :, H, :] = 0.0   # hook_z: [batch, pos, head, d_head]
                    return z
                return hook
            hooks.append((f"blocks.{L}.attn.hook_z", mk(Hs)))
        return hooks

    # ====================================================================
    # the eval: for a given edit (target-prompt hook + sibling-prompt hook builders), measure
    # target-suppression + sibling-collateral over the eval set.
    # ====================================================================
    STOPW = set()
    def measure(hook_target, hook_sib, label=""):
        """hook_target(ix) -> fwd_hooks for the TARGET prompt; hook_sib(ix, sib) -> fwd_hooks for a sibling
        prompt. Returns (target_suppression, sibling_collateral, sib_retention_abs, degenerate_frac)."""
        t_supp = []; s_coll = []; degen = 0; degen_n = 0
        for ix in EVAL:
            it, E, V, qi, ans_tid, ids, tt, qpos, kE, kV = inst_prompt(ix)
            base_t = it.get("base_p") or p_at(tt, ans_tid)
            h = hook_target(ix, tt, qpos, kE, kV)
            cut_t = torch.softmax(model.run_with_hooks(tt, fwd_hooks=h)[0][qpos].float(), -1)[ans_tid].item() if h else base_t
            if base_t > 1e-6:
                t_supp.append(max(0.0, (base_t - cut_t) / base_t))
            # degenerate guard: did the edited answer become a non-name garbage argmax?
            if h:
                am = int(model.run_with_hooks(tt, fwd_hooks=h)[0][qpos].argmax().item())
                degen_n += 1
                if am not in name_tid.values(): degen += 1
            # siblings
            sibs = sibling_prompts(ix)
            for sib in sibs:
                if sib["base"] <= 1e-6: continue
                hs = hook_sib(ix, sib)
                cut_s = (torch.softmax(model.run_with_hooks(sib["tt"], fwd_hooks=hs)[0][sib["qpos"]].float(), -1)[sib["sib_tid"]].item()
                         if hs else sib["base"])
                s_coll.append(max(0.0, (sib["base"] - cut_s) / sib["base"]))
        return (float(np.mean(t_supp)) if t_supp else 0.0,
                float(np.mean(s_coll)) if s_coll else 0.0,
                float(degen / degen_n) if degen_n else 0.0)

    # ====================================================================
    # STAGE 2: FRA cell-cut curve (sweep c) + the oracle hard-cut ceiling
    #   cache the FRA delta per eval/sibling prompt on first use.
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 2 FRA cell-cut curve (heads={FRA_HEADS})")
    _fra_cache = {}
    def fra_delta_target(ix, tt, qpos, kE, kV):
        if ix not in _fra_cache:
            _fra_cache[ix] = build_fra_delta(tt, qpos, kE, kV)
        return _fra_cache[ix]
    _fra_sib_cache = {}
    def fra_delta_sib(ix, sib):
        key = (ix, sib["j"])
        if key not in _fra_sib_cache:
            _fra_sib_cache[key] = build_fra_delta(sib["tt"], sib["qpos"], sib["kE"], sib["kV"])
        return _fra_sib_cache[key]

    done_c = set(p["c"] for p in state["fra_curve"])
    for c in FRA_CS:
        if c in done_c: continue
        def ht(ix, tt, qpos, kE, kV, _c=c):
            byL = fra_delta_target(ix, tt, qpos, kE, kV); return fra_hooks(byL, _c)
        def hs(ix, sib, _c=c):
            byL = fra_delta_sib(ix, sib); return fra_hooks(byL, _c)
        ts, sc, dg = measure(ht, hs, f"FRA c={c}")
        state["fra_curve"].append(dict(c=c, target_supp=ts, sib_coll=sc, degen=dg)); ckpt()
        log(f"  FRA c={c:5.1f}: target_supp={ts:.3f} sib_coll={sc:.4f} degen={dg:.3f}")
    # oracle hard-cut ceiling (c=inf)
    if state.get("fra_oracle") is None:
        def ht(ix, tt, qpos, kE, kV): return hard_cut_hooks(FRA_HEADS, qpos, kE, kV)
        def hs(ix, sib): return hard_cut_hooks(FRA_HEADS, sib["qpos"], sib["kE"], sib["kV"])
        ts, sc, dg = measure(ht, hs, "FRA oracle")
        state["fra_oracle"] = dict(target_supp=ts, sib_coll=sc, degen=dg); ckpt()
        log(f"  FRA ORACLE (hard cut): target_supp={ts:.3f} sib_coll={sc:.4f} degen={dg:.3f}")

    # ====================================================================
    # STAGE 3: build the DoM "retrieve-THIS-bound-value" vector on the TRAIN split
    #   v_L = mean(resid_L | target answer-pos, target prompt) - mean(resid_L | SAME prompt, target value
    #   SWAPPED with another value so the answer changes) -> the direction that retrieves THIS binding's value.
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 3 build DoM vectors on TRAIN (layers {LIN_LAYERS})")
    if state.get("dom_vecs") is None:
        pos = {L: [] for L in LIN_LAYERS}; neg = {L: [] for L in LIN_LAYERS}
        hk = {L: f"blocks.{L}.hook_resid_pre" for L in LIN_LAYERS}
        for ix in TRAIN:
            it = kept[ix]; E, V, qi = it["E"], it["V"], it["qi"]
            prompt, ans = render(E, V, qi)
            _, ttp = build_tokens(prompt)
            cp = model.run_with_cache(ttp, names_filter=list(hk.values()))[1]
            for L in LIN_LAYERS: pos[L].append(cp[hk[L]][0, -1].float())
            # contrast: swap the QUERIED value with a different one present, so the queried binding changes
            Vc = list(V); other = [v for v in values if v not in V]
            if other:
                Vc[qi] = other[0]
            else:
                Vc[qi] = V[(qi + 1) % N_BIND]
            promptc, _ = render(E, Vc, qi)
            _, ttc = build_tokens(promptc)
            cc = model.run_with_cache(ttc, names_filter=list(hk.values()))[1]
            for L in LIN_LAYERS: neg[L].append(cc[hk[L]][0, -1].float())
        vecs = {}
        for L in LIN_LAYERS:
            v = torch.stack(pos[L]).mean(0) - torch.stack(neg[L]).mean(0)
            v = v / (v.norm() + 1e-6)
            vecs[L] = v.cpu().tolist()
        state["dom_vecs"] = vecs; ckpt()
        log(f"  built DoM vectors for layers {LIN_LAYERS}")
    DOM = {int(L): torch.tensor(v, device=dev, dtype=torch.float16) for L, v in state["dom_vecs"].items()}

    # ====================================================================
    # STAGE 4: linear DoM curve (layers x alpha)
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 4 linear DoM sweep (layers {LIN_LAYERS} x alpha {LIN_ALPHA})")
    done_lin = set((p["layer"], p["alpha"]) for p in state["lin_curve"])
    for L in LIN_LAYERS:
        vhat = DOM[L]
        for a in LIN_ALPHA:
            if (L, a) in done_lin: continue
            def ht(ix, tt, qpos, kE, kV, _L=L, _v=vhat, _a=a): return dom_hooks(_L, _v, _a)
            def hs(ix, sib, _L=L, _v=vhat, _a=a): return dom_hooks(_L, _v, _a)
            ts, sc, dg = measure(ht, hs, f"LIN L{L} a={a}")
            state["lin_curve"].append(dict(layer=L, alpha=a, target_supp=ts, sib_coll=sc, degen=dg)); ckpt()
            log(f"  LIN L{L} a={a:5.1f}: target_supp={ts:.3f} sib_coll={sc:.4f} degen={dg:.3f}")

    # ====================================================================
    # STAGE 5: head-ablation reference (full ablation -> single point per head-set)
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 5 head-ablation reference")
    if not state["headabl"]:
        for tag, hs_heads in [("L22H4", ABL_HEADS_A), ("L22H4+L18H6", ABL_HEADS_B)]:
            def ht(ix, tt, qpos, kE, kV, _h=hs_heads): return headabl_hooks(_h)
            def hs(ix, sib, _h=hs_heads): return headabl_hooks(_h)
            ts, sc, dg = measure(ht, hs, f"ABL {tag}")
            state["headabl"].append(dict(tag=tag, target_supp=ts, sib_coll=sc, degen=dg)); ckpt()
            log(f"  ABL {tag}: target_supp={ts:.3f} sib_coll={sc:.4f} degen={dg:.3f}")

    # ====================================================================
    # STAGE 6: matched-point comparison @ t* + verdict
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 6 matched-point @ t*={T_STAR}")
    def interp_coll(curve, t):
        xs = [p["target_supp"] for p in curve]; ys = [p["sib_coll"] for p in curve]
        if not xs or max(xs) < t - 1e-9: return None
        o = np.argsort(xs)
        return float(np.interp(t, np.array(xs)[o], np.array(ys)[o]))

    fra_curve = state["fra_curve"] + [dict(c=1e9, **state["fra_oracle"])] if state.get("fra_oracle") else state["fra_curve"]
    fra_max_supp = max([p["target_supp"] for p in fra_curve], default=0.0)
    # if FRA can't reach T_STAR, drop t* to FRA's reachable max (the reachability lesson)
    t_star = T_STAR if fra_max_supp >= T_STAR else max(0.5, fra_max_supp - 0.02)
    fra_coll = interp_coll(fra_curve, t_star)
    # best-tuned linear: per layer, the curve reaching t* with the LOWEST collateral
    lin_by_layer = {L: [p for p in state["lin_curve"] if p["layer"] == L] for L in LIN_LAYERS}
    lin_choice = None
    for L in LIN_LAYERS:
        coll = interp_coll(lin_by_layer[L], t_star)
        if coll is None: continue
        if (lin_choice is None) or (coll < lin_choice[1]):
            lin_choice = (L, coll)
    lin_layer = lin_choice[0] if lin_choice else None
    lin_coll = lin_choice[1] if lin_choice else None
    lin_max_supp = max([p["target_supp"] for p in state["lin_curve"]], default=0.0)
    # head-ablation: report both; use the one nearest t* (full ablation is a fixed point)
    abl_report = state["headabl"]

    def ratio(lin, fra):
        if lin is None or fra is None: return None
        if fra <= 1e-9: return float("inf") if lin > 1e-9 else 1.0
        return lin / fra
    headline = ratio(lin_coll, fra_coll)
    # head-ablation ratio at its own suppression (compare collateral at matched-ish supp)
    abl_ratios = {a["tag"]: (a["sib_coll"], ratio(a["sib_coll"], fra_coll)) for a in abl_report}

    fra_reaches = fra_coll is not None
    abs_adv = (lin_coll - fra_coll) if (lin_coll is not None and fra_coll is not None) else None
    fra_beats_abl = all((fra_coll is not None and fra_coll < a["sib_coll"] + 1e-9) for a in abl_report) if abl_report else False
    win = bool(fra_reaches and headline is not None and headline >= 2.0
               and abs_adv is not None and abs_adv >= 0.15
               and (lin_coll is not None and fra_coll is not None and fra_coll < lin_coll)
               and fra_beats_abl)
    verdict = "WIN" if win else "NULL"

    summary = dict(
        model=MODEL_NAME, t_star_requested=T_STAR, t_star_used=t_star,
        eval_n=len(EVAL), train_n=len(TRAIN),
        fra_heads=FRA_HEADS,
        fra_max_supp=fra_max_supp, lin_max_supp=lin_max_supp,
        matched_point=dict(
            FRA=dict(sib_collateral=fra_coll, t_star=t_star),
            LINEAR=dict(layer=lin_layer, sib_collateral=lin_coll),
            HEAD_ABLATION={a["tag"]: dict(target_supp=a["target_supp"], sib_collateral=a["sib_coll"]) for a in abl_report},
        ),
        headline_ratio_linear_over_fra=headline,
        abl_ratios={k: dict(sib_coll=v[0], ratio_over_fra=v[1]) for k, v in abl_ratios.items()},
        abs_sib_collateral_advantage=abs_adv,
        fra_curve=fra_curve, lin_curve_best_layer=lin_by_layer.get(lin_layer),
        VERDICT=verdict,
    )
    state["summary"] = summary; ckpt()
    json.dump(summary, open(os.path.join(OUT, "binding_selectivity_summary.json"), "w"), indent=2, default=float)

    log("\n================ BINDING §3.3 SELECTIVITY SUMMARY ================")
    log(f"  model: {MODEL_NAME}   eval n={len(EVAL)} train n={len(TRAIN)}")
    log(f"  matched target-suppression t* = {t_star:.3f} (requested {T_STAR}; FRA max supp {fra_max_supp:.3f}, LIN max {lin_max_supp:.3f})")
    log(f"  FRA  (L22H4+L18H6) sibling-collateral @ t*: {fra_coll}")
    log(f"  LINEAR (best L{lin_layer}) sibling-collateral @ t*: {lin_coll}")
    for a in abl_report:
        log(f"  HEAD-ABL {a['tag']}: target_supp={a['target_supp']:.3f} sibling-collateral={a['sib_coll']:.4f}")
    log(f"  >>> HEADLINE sibling-collateral RATIO (linear/FRA) @ t*: {headline}  (abs adv {abs_adv})")
    log(f"  >>> VERDICT: {verdict} <<<")
    log("DONE binding_selectivity")

except SystemExit:
    raise
except Exception as e:
    tb = traceback.format_exc()
    log("FATAL: " + str(e)); log(tb)
    try:
        json.dump({"FATAL": str(e), "traceback": tb, "t": time.strftime("%H:%M:%S")},
                  open(os.path.join(OUT, "FATAL.json"), "w"), indent=2)
    except Exception:
        pass
    sys.exit(1)
