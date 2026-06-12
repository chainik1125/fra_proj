"""IN-CONTEXT VARIABLE BINDING (A1, Mixing-Mechanisms bound-entity retrieval) — the OPERATING-REGIME
PRE-CHECK (SCREEN2 §3.2 / PREDICTOR2 §4 Step 0). One pod, judge-free, the spend-router for the selectivity
win-test. NO selectivity here.

Re-points the v1 generic FRA cell-cut harness (injection_precheck.py + injection_prefill.py): residual
gemma-scope-2b-pt-res-canonical -> query/key feature decode via W_dec -> per-head pre-softmax score-delta
subtracted at blocks.{L}.attn.hook_attn_scores. The answer-token query replaces injection's response
position; the TARGET binding's entity/value span key replaces the injected-span key.

THE EVAL (self-built, judge-free, ground-truth next-token exact-match):
  template:  "{E0} has the {V0}, {E1} has the {V1}, {E2} has the {V2}, {E3} has the {V3}. Who has the {Vq}?"
  answer:    the entity bound to Vq, as a SINGLE token (" Ann"/" Joe"/...).  GROUND-TRUTH = argmax next tok.
  OPERATING REGIME = keep ONLY instances the model answers CORRECTLY at baseline (binding works). All
  proxies run on this set ONLY (the recall false-GO lesson: sample the app's own operating point).
  P2 = 0 by construction (arbitrary single-token bindings; no fact "Ann has ale" in the weights).

THE FOUR PRE-CHECK GATES (judge-free):
 (a) LOCATE the answer-step retrieve-bound-entity heads: rank all (L,H) by the drop in
     P(correct-answer-first-token | answer pos) when the edge (answer-query -> TARGET entity/value key)
     is zeroed at hook_attn_scores. top<=TOPH heads. (= injection STAGE 2 causal_inj_heads, re-pointed.)
 (b) P1 — oracle answer-step edge-cut R, SPLIT R_gen (cut at the answer/decode query row only) vs
     R_prefill (cut over prefill query rows only). The answer-step read must be load-bearing:
     R_gen >= 0.6 (GO) / >= 0.8 (STRONG). R_gen<0.3 while R_prefill=1 -> upstream/redundant -> NO-GO.
 (c) P3 — SIBLING-BLEED: apply the SAME (target entity x target value) cell-cut, then re-query the SAME
     prompt for the OTHER bindings. bleed = mean(sibling P(correct)-drop) / target P(correct)-drop.
     bleed < 15% (GO) / < 5% (STRONG). bleed > 40% -> slot/role-keyed -> NO-GO.
 (d) ORDER-SHUFFLE CONTENT-VS-POSITION AUDIT (the decisive falsifier): SHUFFLE binding order so the
     target entity's value moves to a DIFFERENT slot while the entity still binds it by CONTENT. Two cuts:
       - CONTENT cell (target-entity x target-value tokens, wherever they now sit) -> if it still
         suppresses the target's retrieval -> CONTENT-addressed -> WIN-consistent.
       - POSITION cell (the slot the target ORIGINALLY occupied, now holding a DIFFERENT binding) ->
         if THAT suppresses the target's retrieval -> POSITION-addressed -> box ~1.9x floor -> NO-GO.
     GATE: edge must be CONTENT-addressed (content-cut suppression > position-cut suppression of target).

VERDICT — GO iff R_gen >= 0.6 AND sibling-bleed < 15% AND edge is content-addressed. NO-GO otherwise =
a clean one-pod pre-registered negative. Report all four numbers regardless.

Resume-proof: an early flush print + ckpt() BEFORE any heavy loop, ckpt() INSIDE every per-head /
per-instance loop, and a top-of-script try/except that uploads the traceback (the v1 restart-loop fix).
"""
import os, sys, json, time, traceback
sys.path.insert(0, "/workspace/code")

OUT = os.environ.get("OUTDIR", ".")
CKPT = os.path.join(OUT, "binding_precheck.json")

def log(*a):
    print(*a, flush=True)

# ---- EARLY heartbeat + flush BEFORE any heavy import/loop (v1 monolithic-ckpt blindness fix) ----
log(f"[{time.strftime('%H:%M:%S')}] PYJOB START — binding_precheck entered (OUT={OUT})")
try:
    os.makedirs(OUT, exist_ok=True)
    json.dump({"heartbeat": "entered", "t": time.strftime("%H:%M:%S")},
              open(os.path.join(OUT, "heartbeat.json"), "w"))
except Exception:
    pass

# top-of-script try/except: any crash uploads the traceback to OUT so the orchestrator sees WHY
try:
    import torch, numpy as np
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    from fra.sae_lens_wrapper import GemmaScopeSAE
    from fra.core.fra import _build_fra_result

    # ---- config ---------------------------------------------------------
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-2-2b-it")
    N_INST     = int(os.environ.get("N_INST", "200"))   # candidate instances to build
    N_TARGET   = int(os.environ.get("N_TARGET", "40"))  # cap on operating-regime instances to run proxies
    TOPH       = int(os.environ.get("TOPH", "3"))       # <=3 causal retrieve-bound-entity heads
    M_PAIRS    = int(os.environ.get("M_PAIRS", "16"))   # top SAE feature-pairs per head edge (kept for parity)
    N_BIND     = 4                                       # bindings per instance
    SEED       = int(os.environ.get("SEED", "0"))

    def ckpt():
        json.dump(state, open(CKPT, "w"), indent=2, default=float)

    log(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=dev, dtype=torch.float16)
    model.eval()
    tok = model.tokenizer
    NL = model.cfg.n_layers
    NH = model.cfg.n_heads
    log(f"  loaded: {NL} layers x {NH} heads")

    # ====================================================================
    # VOCAB — single-token entities + values (verified at runtime on THIS tokenizer)
    # ====================================================================
    # Entities (names) and values (short objects). We keep only those that tokenize to exactly ONE token
    # WITH a leading space (the answer position emits " Ann" / " ale"), so next-token exact-match is clean.
    NAME_POOL = ["Ann", "Joe", "Pete", "Tim", "Sam", "Tom", "Kate", "Mark", "Paul", "Jane",
                 "Bob", "Sue", "Dan", "Lucy", "Mike", "Anna", "Jack", "Mary", "Ben", "Rose",
                 "Carl", "Nina", "Eric", "Lisa", "Gary", "Emma", "Fred", "Ruth", "Adam", "Beth"]
    VALUE_POOL = ["ale", "pie", "jam", "tea", "ham", "fig", "cod", "oat", "bun", "egg",
                  "nut", "yam", "cake", "wine", "fish", "rice", "soup", "milk", "corn", "plum",
                  "pear", "lime", "kale", "beef", "duck", "lamb", "crab", "rye", "oil", "bread"]

    def single_tok(word):
        """True iff ' '+word encodes to exactly one token. Returns (ok, first_tid)."""
        ids = tok.encode(" " + word, add_special_tokens=False)
        return (len(ids) == 1), (ids[0] if ids else None)

    names, name_tid = [], {}
    for w in NAME_POOL:
        ok, t = single_tok(w)
        if ok:
            names.append(w); name_tid[w] = t
    values, value_tid = [], {}
    for w in VALUE_POOL:
        ok, t = single_tok(w)
        if ok:
            values.append(w); value_tid[w] = t
    log(f"  single-token pool: {len(names)} names, {len(values)} values")
    log(f"    names={names}")
    log(f"    values={values}")
    if len(names) < N_BIND or len(values) < N_BIND:
        raise RuntimeError(f"too few single-token vocab: names={len(names)} values={len(values)}")

    # ====================================================================
    # BUILD instances (N_INST), each with 4 distinct (entity,value) bindings + a query value
    # ====================================================================
    rng = np.random.RandomState(SEED)
    def build_instance():
        E = list(rng.choice(names, N_BIND, replace=False))
        V = list(rng.choice(values, N_BIND, replace=False))
        qi = int(rng.randint(N_BIND))     # which binding is queried
        return dict(E=E, V=V, qi=qi)

    def render(E, V, qi):
        """prompt text; queried by VALUE -> answer is the ENTITY (single token). 'Who has the {Vq}?'
        COMPLETION format ending in ' Answer:' so the IMMEDIATE next token is the bound entity ' Ann'/...
        (the chat-template + add_generation_prompt format made the model emit a preamble token first ->
        p_correct~=0 at the answer position; run-1 BLOCKER. The Mixing-Mechanisms / SCREEN2 '->' intent.)"""
        parts = [f"{E[i]} has the {V[i]}" for i in range(N_BIND)]
        body = ", ".join(parts) + "."
        prompt = f"{body} Who has the {V[qi]}? Answer:"
        answer = E[qi]                     # bound entity, emitted as the next token
        return prompt, answer

    def chat_wrap(user_text):
        # COMPLETION mode: no chat template (it forced a preamble token before the answer). Identity.
        return user_text
    def build_tokens(text):
        ids = tok.encode(text, add_special_tokens=True)    # prepend <bos> (no chat template now)
        return ids, torch.tensor(ids, device=dev).unsqueeze(0)

    def p_correct(tt, ans_tid, qpos=None):
        """P(answer first token | answer position). qpos default = last."""
        logits = model(tt)[0]
        q = (tt.shape[1] - 1) if qpos is None else qpos
        return torch.softmax(logits[q].float(), -1)[ans_tid].item()

    def argmax_tid(tt, qpos=None):
        logits = model(tt)[0]
        q = (tt.shape[1] - 1) if qpos is None else qpos
        return int(logits[q].argmax().item())

    # ---- token-span finder: locate the token index of a single-token word before pos q ----
    def find_word_pos(ids, word, q, occurrence=0):
        """index of the `occurrence`-th token of ' '+word (single token) in ids[:q]. None if absent."""
        wt = tok.encode(" " + word, add_special_tokens=False)
        if len(wt) != 1:
            return None
        t = wt[0]
        hits = [i for i in range(q) if ids[i] == t]
        if occurrence < len(hits):
            return hits[occurrence]
        return None

    # ====================================================================
    # resume state
    # ====================================================================
    state = {"config": {"model": MODEL_NAME, "topH": TOPH, "M_pairs": M_PAIRS, "n_bind": N_BIND,
                        "n_inst": N_INST, "n_target": N_TARGET, "seed": SEED,
                        "sae": "gemma-scope-2b-pt-res-canonical"},
             "instances": [], "baseline": [], "opset": None,
             "heads": None, "head_effects": None,
             "p1": [], "p3": [], "shuffle": []}
    if os.path.exists(CKPT):
        try:
            state = json.load(open(CKPT))
            log(f"  resumed: baseline={len(state['baseline'])} p1={len(state['p1'])} "
                f"p3={len(state['p3'])} shuffle={len(state['shuffle'])} heads={state.get('heads')}")
        except Exception:
            pass
    ckpt()

    # ====================================================================
    # SCORE-CELL CUT machinery (= injection cut_hooks / make_cut_hooks, re-pointed)
    # ====================================================================
    def cut_hooks_timing(heads, qpos, kpos, prompt_len, timing):
        """zero (q,k)=(query-rows, kpos) at hook_attn_scores at `heads`.
        timing in {'all','gen','pre'}: which query rows are blocked.
          all = every query row; gen = only the answer/decode row (q==qpos, i.e. the answer step);
          pre = only prompt-internal rows (q < qpos). kpos is the TARGET key token index."""
        byL = {}
        for L, H in heads:
            byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    seq = s.shape[2]
                    if kpos >= seq:
                        return s
                    qq = qpos if qpos < seq else seq - 1
                    if timing == "all":
                        rows = slice(0, seq)
                    elif timing == "gen":
                        rows = slice(qq, qq + 1)       # the answer step ONLY
                    else:  # pre
                        rows = slice(0, qq)             # prefill rows ONLY
                    for H in Hs:
                        s[0, H, rows, kpos] = -1e4
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return hooks

    def p_correct_cut(tt, ans_tid, heads, qpos, kpos, timing):
        h = cut_hooks_timing(heads, qpos, kpos, tt.shape[1], timing)
        logits = model.run_with_hooks(tt, fwd_hooks=h)[0]
        return torch.softmax(logits[qpos].float(), -1)[ans_tid].item()

    # ====================================================================
    # STAGE 1: BUILD + BASELINE filter -> operating regime (baseline-correct)
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 1 build {N_INST} instances + baseline-correct filter")
    if not state["instances"]:
        insts = [build_instance() for _ in range(N_INST)]
        # de-dup on (E,V,qi)
        seen = set(); uniq = []
        for it in insts:
            k = (tuple(it["E"]), tuple(it["V"]), it["qi"])
            if k not in seen:
                seen.add(k); uniq.append(it)
        state["instances"] = uniq
        ckpt()
    insts = state["instances"]
    log(f"  built {len(insts)} unique candidate instances")

    seen_b = set(r["idx"] for r in state["baseline"])
    for ix, it in enumerate(insts):
        if ix in seen_b:
            continue
        try:
            prompt, ans = render(it["E"], it["V"], it["qi"])
            ans_tid = name_tid[ans]
            ids, tt = build_tokens(chat_wrap(prompt))
            if tt.shape[1] > 200:
                continue
            am = argmax_tid(tt)
            pc = p_correct(tt, ans_tid)
            correct = (am == ans_tid)
            state["baseline"].append(dict(idx=ix, correct=bool(correct), p_correct=float(pc),
                                          answer=ans, plen=int(tt.shape[1])))
            if (len(state["baseline"]) % 10) == 0:
                ckpt()
                log(f"  [{ix:3d}] correct={correct} p_correct={pc:.3f} ans='{ans}'")
        except Exception as e:
            log(f"  [base {ix}] ERR {e}")
            state.setdefault("errors", []).append("s1:" + str(e)[:160])
            ckpt()
    ckpt()

    opset = [r["idx"] for r in state["baseline"] if r["correct"]]
    state["opset"] = opset[:N_TARGET]
    state["baseline_acc"] = len(opset) / max(1, len(state["baseline"]))
    ckpt()
    log(f"  OPERATING REGIME: {len(opset)} baseline-correct / {len(state['baseline'])} tried "
        f"(acc={state['baseline_acc']:.2f}); running proxies on first {len(state['opset'])}")

    if len(state["opset"]) < 8:
        state["BLOCKER"] = (f"too few baseline-correct binding instances ({len(state['opset'])}); "
                            f"gemma-2-2b-it binding may fail at 4 bindings. Try N_BIND=3 or easier vocab.")
        ckpt()
        json.dump(state, open(os.path.join(OUT, "binding_summary.json"), "w"), indent=2, default=float)
        log("BLOCKER: " + state["BLOCKER"])
        sys.exit(0)

    OP = state["opset"]

    # ====================================================================
    # STAGE 2: LOCATE the retrieve-bound-entity heads (= injection STAGE 2 head-find, re-pointed)
    #   target binding = the QUERIED binding (E[qi] x V[qi]); key = the TARGET ENTITY token (E[qi])
    #   in the binding list. (Entity is the answer content; we cut the answer-query -> entity-key edge.)
    # ====================================================================
    if state.get("heads") is None:
        log(f"[{time.strftime('%H:%M:%S')}] STAGE 2 head-find (answer-query -> target-entity key)")
        # use up to 4 anchors for a robust per-head ranking (mean drop in P(correct))
        anchors = OP[:min(4, len(OP))]
        dd = {(L, H): 0.0 for L in range(NL) for H in range(NH)}
        n_anch = 0
        for ix in anchors:
            it = insts[ix]
            prompt, ans = render(it["E"], it["V"], it["qi"])
            ans_tid = name_tid[ans]
            ids, tt = build_tokens(chat_wrap(prompt))
            qpos = tt.shape[1] - 1
            # TARGET key = the entity token E[qi] in the BINDING LIST (first occurrence = its binding slot)
            kpos = find_word_pos(ids, ans, qpos, occurrence=0)
            if kpos is None:
                continue
            base = p_correct(tt, ans_tid)
            for L in range(NL):
                for H in range(NH):
                    pc = p_correct_cut(tt, ans_tid, [(L, H)], qpos, kpos, "all")
                    dd[(L, H)] += (base - pc)
            n_anch += 1
            ckpt()  # ckpt INSIDE the per-anchor loop (heavy)
            log(f"  head-find anchor {ix} done (base P(correct)={base:.3f}, kpos={kpos})")
        if n_anch == 0:
            raise RuntimeError("head-find: no anchor had a locatable target-entity key")
        ddm = {k: v / n_anch for k, v in dd.items()}
        top = sorted(ddm, key=lambda x: -ddm[x])[:TOPH]
        state["heads"] = [list(h) for h in top]
        state["head_effects"] = {f"L{L}H{H}": float(ddm[(L, H)]) for (L, H) in top}
        ckpt()
        log(f"  top-{TOPH} retrieve-bound-entity heads: {state['head_effects']}")
    heads = [tuple(h) for h in state["heads"]]

    # ====================================================================
    # STAGE 3: P1 — oracle answer-step edge-cut R, SPLIT R_gen vs R_prefill
    #   cut (answer-query x target-entity-key) at top heads; measure P(correct)-drop fraction.
    #   R_gen = cut only the answer row; R_prefill = cut only prefill rows; R_all = every row.
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 3 P1 answer-step R_gen/R_prefill (heads={heads})")
    done_p1 = set(r["idx"] for r in state["p1"])
    for ix in OP:
        if ix in done_p1:
            continue
        try:
            it = insts[ix]
            prompt, ans = render(it["E"], it["V"], it["qi"])
            ans_tid = name_tid[ans]
            ids, tt = build_tokens(chat_wrap(prompt))
            qpos = tt.shape[1] - 1
            kpos = find_word_pos(ids, ans, qpos, occurrence=0)   # target entity token in the list
            if kpos is None:
                continue
            base = p_correct(tt, ans_tid)
            p_all = p_correct_cut(tt, ans_tid, heads, qpos, kpos, "all")
            p_gen = p_correct_cut(tt, ans_tid, heads, qpos, kpos, "gen")
            p_pre = p_correct_cut(tt, ans_tid, heads, qpos, kpos, "pre")
            state["p1"].append(dict(idx=ix, base=float(base), p_all=float(p_all),
                                    p_gen=float(p_gen), p_pre=float(p_pre), kpos=int(kpos)))
            ckpt()  # ckpt INSIDE the per-instance loop
            log(f"  p1 [{ix:3d}] base={base:.3f} all={p_all:.3f} gen={p_gen:.3f} pre={p_pre:.3f}")
        except Exception as e:
            log(f"  p1 ERR {e}\n{traceback.format_exc()[:200]}")
            state.setdefault("errors", []).append("s3:" + str(e)[:160])
            ckpt()

    def frac_drop(rows, base_key, cut_key):
        """mean fractional drop in P(correct): mean( (base-cut)/base )."""
        ds = []
        for r in rows:
            b = r[base_key]
            if b > 1e-6:
                ds.append(max(0.0, (b - r[cut_key]) / b))
        return float(np.mean(ds)) if ds else None

    P1 = state["p1"]
    R_all = frac_drop(P1, "base", "p_all")
    R_gen = frac_drop(P1, "base", "p_gen")
    R_pre = frac_drop(P1, "base", "p_pre")
    state["R"] = dict(all=R_all, gen=R_gen, prefill=R_pre, n=len(P1))
    ckpt()
    log(f"  P1: R_all={R_all} R_gen={R_gen} R_prefill={R_pre}  (n={len(P1)})")

    # ====================================================================
    # STAGE 4: P3 — SIBLING-BLEED. Apply the SAME (target entity x target value) cell-cut, then
    #   re-query the SAME prompt for the OTHER bindings. We cut the (answer-query -> target-entity)
    #   AND (answer-query -> target-value) edges (the target conjunction), then ask each sibling
    #   "Who has the {V[j]}?" on the same context and measure the drop in the sibling's P(correct).
    #   bleed = mean(sibling fractional drop) / target fractional drop.
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 4 P3 sibling-bleed (same cut, re-query other bindings)")
    done_p3 = set(r["idx"] for r in state["p3"])

    def cut_hooks_pair(heads, qpos, kE, kV):
        """zero (answer-query -> target-entity) AND (answer-query -> target-value) at heads, at the
        answer step (the conjunction cut). Applied at all rows for robustness with answer-row q tracked."""
        byL = {}
        for L, H in heads:
            byL.setdefault(L, []).append(H)
        hooks = []
        for L, Hs in byL.items():
            def mk(Hs):
                def hook(s, hook):
                    seq = s.shape[2]
                    qq = qpos if qpos < seq else seq - 1
                    for H in Hs:
                        if kE < seq:
                            s[0, H, qq, kE] = -1e4
                        if kV < seq:
                            s[0, H, qq, kV] = -1e4
                    return s
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(Hs)))
        return hooks

    for ix in OP:
        if ix in done_p3:
            continue
        try:
            it = insts[ix]
            E, V, qi = it["E"], it["V"], it["qi"]
            ans = E[qi]; ans_tid = name_tid[ans]
            # build the TARGET prompt (query the target's value) and locate target entity+value keys
            prompt_t, _ = render(E, V, qi)
            ids_t, tt_t = build_tokens(chat_wrap(prompt_t))
            qpos_t = tt_t.shape[1] - 1
            kE = find_word_pos(ids_t, E[qi], qpos_t, occurrence=0)
            kV = find_word_pos(ids_t, V[qi], qpos_t, occurrence=0)   # first occ = the binding-list value
            if kE is None or kV is None:
                continue
            base_t = p_correct(tt_t, ans_tid)
            h_t = cut_hooks_pair(heads, qpos_t, kE, kV)
            cut_t = torch.softmax(model.run_with_hooks(tt_t, fwd_hooks=h_t)[0][qpos_t].float(), -1)[ans_tid].item()
            target_drop = max(0.0, (base_t - cut_t) / base_t) if base_t > 1e-6 else 0.0

            # SIBLINGS: same context, query each OTHER binding's value; cut the SAME (target E x target V) cell.
            sib = []
            for j in range(N_BIND):
                if j == qi:
                    continue
                prompt_s, _ = render(E, V, j)            # same bindings, query sibling j's value
                ids_s, tt_s = build_tokens(chat_wrap(prompt_s))
                qpos_s = tt_s.shape[1] - 1
                sib_ans = E[j]; sib_tid = name_tid[sib_ans]
                # locate the TARGET entity/value keys in THIS rendering (positions stable: same body)
                kE_s = find_word_pos(ids_s, E[qi], qpos_s, occurrence=0)
                kV_s = find_word_pos(ids_s, V[qi], qpos_s, occurrence=0)
                if kE_s is None or kV_s is None:
                    continue
                base_s = p_correct(tt_s, sib_tid)
                if base_s <= 1e-6:
                    continue
                h_s = cut_hooks_pair(heads, qpos_s, kE_s, kV_s)   # cut the TARGET cell, NOT the sibling's
                cut_s = torch.softmax(model.run_with_hooks(tt_s, fwd_hooks=h_s)[0][qpos_s].float(), -1)[sib_tid].item()
                sib_drop = max(0.0, (base_s - cut_s) / base_s)
                sib.append(dict(j=j, base=float(base_s), cut=float(cut_s), drop=float(sib_drop)))
            mean_sib_drop = float(np.mean([s["drop"] for s in sib])) if sib else None
            state["p3"].append(dict(idx=ix, target_base=float(base_t), target_cut=float(cut_t),
                                    target_drop=float(target_drop), siblings=sib,
                                    mean_sib_drop=mean_sib_drop))
            ckpt()
            log(f"  p3 [{ix:3d}] target_drop={target_drop:.3f} mean_sib_drop={mean_sib_drop}")
        except Exception as e:
            log(f"  p3 ERR {e}\n{traceback.format_exc()[:200]}")
            state.setdefault("errors", []).append("s4:" + str(e)[:160])
            ckpt()

    # bleed = mean sibling drop / mean target drop (across instances), the relative collateral
    P3 = [r for r in state["p3"] if r.get("mean_sib_drop") is not None]
    mean_target_drop = float(np.mean([r["target_drop"] for r in P3])) if P3 else None
    mean_sibling_drop = float(np.mean([r["mean_sib_drop"] for r in P3])) if P3 else None
    sibling_bleed = (mean_sibling_drop / mean_target_drop) if (mean_target_drop and mean_target_drop > 1e-6) else None
    state["P3"] = dict(mean_target_drop=mean_target_drop, mean_sibling_drop=mean_sibling_drop,
                       sibling_bleed=sibling_bleed, n=len(P3))
    ckpt()
    log(f"  P3: target_drop={mean_target_drop} sib_drop={mean_sibling_drop} BLEED={sibling_bleed} (n={len(P3)})")

    # ====================================================================
    # STAGE 5: ORDER-SHUFFLE CONTENT-VS-POSITION AUDIT (the decisive falsifier)
    #   For each instance: ORIGINAL order had target entity E[qi] in slot s0 (=qi). SHUFFLE the binding
    #   order so the target binding (E[qi] x V[qi]) moves to a DIFFERENT slot s1 (entity still binds its
    #   value by CONTENT). On the SHUFFLED prompt (still query V[qi] -> answer E[qi]):
    #     - CONTENT cut: zero (answer-query -> E[qi] token AND V[qi] token wherever they now sit) -> drop_content
    #     - POSITION cut: zero (answer-query -> the entity+value tokens now sitting in the ORIGINAL slot s0,
    #       which now hold a DIFFERENT binding) -> drop_position
    #   CONTENT-addressed iff drop_content suppresses the target MORE than drop_position.
    # ====================================================================
    log(f"[{time.strftime('%H:%M:%S')}] STAGE 5 order-shuffle content-vs-position audit")
    done_sh = set(r["idx"] for r in state["shuffle"])
    for ix in OP:
        if ix in done_sh:
            continue
        try:
            it = insts[ix]
            E, V, qi = it["E"], it["V"], it["qi"]
            ans = E[qi]; ans_tid = name_tid[ans]
            s0 = qi                                  # the target's ORIGINAL slot
            # build a permutation that MOVES the target binding to a different slot
            perm = list(range(N_BIND))
            rng2 = np.random.RandomState(1000 + ix)
            for _try in range(20):
                rng2.shuffle(perm)
                if perm.index(s0) != s0:             # target binding lands in a new slot
                    break
            s1 = perm.index(s0)                      # the target's NEW slot
            if s1 == s0:
                continue
            # shuffled bindings: slot k holds binding perm[k]
            Esh = [E[perm[k]] for k in range(N_BIND)]
            Vsh = [V[perm[k]] for k in range(N_BIND)]
            qi_sh = s1                               # target binding now sits at slot s1
            prompt_sh, ans_sh = render(Esh, Vsh, qi_sh)   # query V[qi] (==Vsh[s1]); answer E[qi]
            assert ans_sh == ans, f"shuffle answer mismatch {ans_sh} vs {ans}"
            ids_sh, tt_sh = build_tokens(chat_wrap(prompt_sh))
            qpos_sh = tt_sh.shape[1] - 1
            base_sh = p_correct(tt_sh, ans_tid)
            if base_sh <= 1e-6:
                # model no longer answers correctly under shuffle -> skip (off operating regime)
                state["shuffle"].append(dict(idx=ix, base_sh=float(base_sh), skipped="base0"))
                ckpt(); continue
            # CONTENT cut: target entity+value tokens, wherever they now sit (slot s1)
            kE_c = find_word_pos(ids_sh, E[qi], qpos_sh, occurrence=0)
            kV_c = find_word_pos(ids_sh, V[qi], qpos_sh, occurrence=0)
            # POSITION cut: the entity+value now occupying the ORIGINAL slot s0 (a DIFFERENT binding)
            E_pos = Esh[s0]; V_pos = Vsh[s0]
            kE_p = find_word_pos(ids_sh, E_pos, qpos_sh, occurrence=0)
            kV_p = find_word_pos(ids_sh, V_pos, qpos_sh, occurrence=0)
            if None in (kE_c, kV_c, kE_p, kV_p):
                continue
            h_c = cut_hooks_pair(heads, qpos_sh, kE_c, kV_c)
            cut_c = torch.softmax(model.run_with_hooks(tt_sh, fwd_hooks=h_c)[0][qpos_sh].float(), -1)[ans_tid].item()
            h_p = cut_hooks_pair(heads, qpos_sh, kE_p, kV_p)
            cut_p = torch.softmax(model.run_with_hooks(tt_sh, fwd_hooks=h_p)[0][qpos_sh].float(), -1)[ans_tid].item()
            drop_content = max(0.0, (base_sh - cut_c) / base_sh)
            drop_position = max(0.0, (base_sh - cut_p) / base_sh)
            state["shuffle"].append(dict(idx=ix, s0=int(s0), s1=int(s1),
                                         base_sh=float(base_sh), cut_content=float(cut_c), cut_position=float(cut_p),
                                         drop_content=float(drop_content), drop_position=float(drop_position)))
            ckpt()
            log(f"  shuffle [{ix:3d}] s0={s0}->s1={s1} drop_content={drop_content:.3f} drop_position={drop_position:.3f}")
        except Exception as e:
            log(f"  shuffle ERR {e}\n{traceback.format_exc()[:200]}")
            state.setdefault("errors", []).append("s5:" + str(e)[:160])
            ckpt()

    SH = [r for r in state["shuffle"] if "drop_content" in r]
    mean_dc = float(np.mean([r["drop_content"] for r in SH])) if SH else None
    mean_dp = float(np.mean([r["drop_position"] for r in SH])) if SH else None
    # per-instance: is content-cut the larger suppressor? fraction content-addressed
    frac_content = (float(np.mean([1.0 if r["drop_content"] > r["drop_position"] else 0.0 for r in SH]))
                    if SH else None)
    content_addressed = (mean_dc is not None and mean_dp is not None and mean_dc > mean_dp
                         and (frac_content is not None and frac_content >= 0.5))
    state["SHUFFLE"] = dict(mean_drop_content=mean_dc, mean_drop_position=mean_dp,
                            frac_content_addressed=frac_content, content_addressed=bool(content_addressed),
                            n=len(SH))
    ckpt()
    log(f"  SHUFFLE: drop_content={mean_dc} drop_position={mean_dp} frac_content={frac_content} "
        f"content_addressed={content_addressed} (n={len(SH)})")

    # ====================================================================
    # VERDICT
    # ====================================================================
    gate_p1 = (R_gen is not None) and (R_gen >= 0.6)
    gate_p3 = (sibling_bleed is not None) and (sibling_bleed < 0.15)
    gate_sh = bool(content_addressed)
    # hard-fail diagnostics
    p1_hardfail = (R_gen is not None and R_pre is not None and R_gen < 0.3 and R_pre >= 0.9)
    p3_hardfail = (sibling_bleed is not None and sibling_bleed > 0.40)
    verdict = "GO" if (gate_p1 and gate_p3 and gate_sh) else "NO-GO"
    strong = (R_gen is not None and R_gen >= 0.8 and sibling_bleed is not None and sibling_bleed < 0.05
              and gate_sh)

    summary = dict(
        model=MODEL_NAME,
        operating_regime_n=len(state["opset"]), baseline_acc=state.get("baseline_acc"),
        baseline_tried=len(state["baseline"]),
        heads=state["heads"], head_effects=state.get("head_effects"),
        P1=dict(R_gen=R_gen, R_prefill=R_pre, R_all=R_all, n=len(P1), hardfail=bool(p1_hardfail)),
        P3=dict(sibling_bleed=sibling_bleed, mean_target_drop=mean_target_drop,
                mean_sibling_drop=mean_sibling_drop, n=len(P3), hardfail=bool(p3_hardfail)),
        SHUFFLE=dict(content_addressed=bool(content_addressed), mean_drop_content=mean_dc,
                     mean_drop_position=mean_dp, frac_content_addressed=frac_content, n=len(SH)),
        gate_p1_R_gen_ge_0p6=bool(gate_p1), gate_p3_bleed_lt_0p15=bool(gate_p3),
        gate_shuffle_content_addressed=bool(gate_sh),
        STRONG=bool(strong), VERDICT=verdict,
    )
    state["summary"] = summary
    ckpt()
    json.dump(summary, open(os.path.join(OUT, "binding_summary.json"), "w"), indent=2, default=float)

    log("\n================ BINDING PRE-CHECK SUMMARY ================")
    log(f"  model: {MODEL_NAME}")
    log(f"  operating regime: {summary['operating_regime_n']} baseline-correct "
        f"(baseline acc={summary['baseline_acc']:.2f} of {summary['baseline_tried']} tried)")
    log(f"  retrieve-bound-entity heads: {summary['heads']}  effects={summary['head_effects']}")
    log(f"  P1: R_gen={R_gen} (GATE >=0.6 -> {gate_p1})  R_prefill={R_pre}  R_all={R_all}")
    log(f"  P3: sibling-bleed={sibling_bleed} (GATE <0.15 -> {gate_p3})  "
        f"[target_drop={mean_target_drop} sib_drop={mean_sibling_drop}]")
    log(f"  SHUFFLE: content_addressed={content_addressed} (GATE -> {gate_sh})  "
        f"[drop_content={mean_dc} drop_position={mean_dp} frac_content={frac_content}]")
    log(f"  >>> VERDICT: {verdict} <<<  (STRONG={strong})")
    log("DONE binding_precheck")

except SystemExit:
    raise
except Exception as e:
    tb = traceback.format_exc()
    log("FATAL: " + str(e))
    log(tb)
    try:
        json.dump({"FATAL": str(e), "traceback": tb, "t": time.strftime("%H:%M:%S")},
                  open(os.path.join(OUT, "FATAL.json"), "w"), indent=2)
    except Exception:
        pass
    sys.exit(1)
