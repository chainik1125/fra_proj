"""RUNG 3 feasibility -- does a planted association fire on SEMANTIC triggers?

RESEARCH CONTEXT. Defensive interpretability research for an academic paper on Feature-Resolved
Attention (FRA), with Dmitry Manning-Coe. Benign placeholder associations only (king -> crown,
dog -> bone). Nothing here intervenes; it only checks that a behaviour exists before any removal
experiment is built. See docs/insen/research_context.md.

Why this exists: rung 2 failed condition 1 ("the behaviour must exist") on both base and IT Gemma,
so the next rung checks the behaviour first, cheaply, with forward passes only.

Question: plant "The password is king crown." (rung 1's template), then query with a DIFFERENT word
for the same concept ("Remember the password: queen"). If P(crown) stays high for same-concept probes
and low for unrelated controls, the trigger is a concept, not a token -- the regime where one
feature-pair cut should disarm every surface form (the "semantic filter" idea).

Reported per concept: P(payload) for the planted word, each same-concept probe, and unrelated controls,
averaged over filler seeds.

Run: python scripts/46_rung3_feasibility.py   (GPU; ~2-5 min; no SAEs)
"""
import os, json
import numpy as np, torch
from transformer_lens import HookedTransformer

OUT = os.environ.get("OUTDIR", "."); os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get("MODEL", "gemma-2-2b")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(MODEL, device=dev, dtype=torch.float16); model.eval(); tok = model.tokenizer
print(f"[model] {MODEL}", flush=True)

FILLER = ["The weather was mild for the time of year.", "Several people arrived late to the meeting.",
 "The library opened an hour earlier than usual.", "A small dog waited patiently by the door.",
 "Nobody expected the report to take so long.", "The train was quiet on the way back.",
 "She wrote a short note and left it on the desk.", "The garden needed more water after the heat.",
 "Two students argued about the final question.", "The coffee in the office had gone cold.",
 "Most of the chairs were stacked against the wall.", "He forgot his umbrella at the station.",
 "The new schedule starts next Monday.", "A light rain began just after lunch.",
 "The shop on the corner closed for repairs.", "They planned a short walk along the lake.",
 "The printer ran out of paper again.", "Everyone agreed the soup was too salty.",
 "The museum added a room for old maps.", "Her sister called to ask about the weekend.",
 "The bus took a longer route through town.", "A few lights were still on in the building.",
 "The children painted a picture of a boat.", "The package arrived two days early.",
 "The team reviewed the notes from last week.", "A cold wind came in from the north.",
 "The old bridge was painted a bright color.", "He read the instructions twice before starting.",
 "The kitchen smelled of fresh bread.", "The road was closed because of the storm."]

# concept: (planted trigger, payload, same-concept probes, unrelated controls)
CONCEPTS = {
    # confirmed concept-level in day-2 feasibility (kept as positive controls)
    "vehicle": (" car",    " garage",  [" truck", " van", " bus", " vehicle"],            [" apple", " song", " chair"]),
    "vessel":  (" ship",   " anchor",  [" boat", " vessel", " yacht", " ships"],          [" pencil", " garden", " cup"]),
    # new candidates for the broader run
    "money":   (" bank",   " vault",   [" money", " cash", " funds", " coins"],           [" leaf", " chair", " cloud"]),
    "weather": (" storm",  " shelter", [" rain", " wind", " thunder", " flood"],          [" desk", " apple", " song"]),
    "fire":    (" fire",   " smoke",   [" flame", " blaze", " fires", " burning"],        [" spoon", " garden", " river"]),
    "music":   (" music",  " concert", [" song", " melody", " tune", " songs"],           [" brick", " window", " potato"]),
    "war":     (" army",   " battle",  [" soldier", " troops", " military", " war"],      [" pillow", " lemon", " chair"]),
    "food":    (" bread",  " oven",    [" cake", " dough", " pastry", " loaf"],            [" rocket", " cloud", " desk"]),
    "school":  (" school", " lesson",  [" teacher", " student", " class", " pupil"],      [" anchor", " cloud", " spoon"]),
    "doctor":  (" doctor", " hospital",[" nurse", " patient", " surgeon", " medicine"],   [" guitar", " river", " brick"]),
    "plant":   (" tree",   " forest",  [" plant", " flower", " leaf", " trees"],          [" engine", " song", " coin"]),
    "bird":    (" bird",   " nest",    [" birds", " sparrow", " robin", " eagle"],        [" hammer", " song", " desk"]),
}
SEEDS = list(range(8))


def prompt_ids(seed, plant, payload, query_word, n_before=4, n_mid=6):
    g = np.random.default_rng(seed)
    fs = [FILLER[i] for i in g.permutation(len(FILLER))]
    text = (" ".join(fs[:n_before]) + f" The password is{plant}{payload}. " + " ".join(fs[n_before:n_before + n_mid])
            + " Remember the password:" + query_word)
    return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)


def single(w):
    return len(tok.encode(w, add_special_tokens=False)) == 1


out = {}
for cname, (plant, payload, probes, controls) in CONCEPTS.items():
    if not (single(plant) and single(payload)):
        print(f"skip {cname}: planted/payload not single-token", flush=True); continue
    pid = tok.encode(payload, add_special_tokens=False)[0]
    res = {}
    for kind, words in (("planted", [plant]), ("same-concept", probes), ("control", controls)):
        for w in words:
            if not single(w):
                print(f"  {cname}: '{w.strip()}' not single-token, skipped", flush=True); continue
            ps = []
            for s in SEEDS:
                ids = prompt_ids(s, plant, payload, w)
                lg = model(torch.tensor(ids, device=dev).unsqueeze(0))[0, -1].float()
                ps.append(torch.softmax(lg, -1)[pid].item())
            res[w.strip()] = dict(kind=kind, mean=float(np.mean(ps)), min=float(np.min(ps)), max=float(np.max(ps)))
    out[cname] = dict(plant=plant.strip(), payload=payload.strip(), probes=res)
    print(f"\n=== {cname}: planted '{plant.strip()}' -> payload '{payload.strip()}' ===", flush=True)
    for w, r in res.items():
        print(f"  {r['kind']:13} {w:10} P(payload) mean {r['mean']:.3f}  [min {r['min']:.3f}, max {r['max']:.3f}]", flush=True)
    sc = [r["mean"] for r in res.values() if r["kind"] == "same-concept"]
    ct = [r["mean"] for r in res.values() if r["kind"] == "control"]
    pl = [r["mean"] for r in res.values() if r["kind"] == "planted"]
    if sc and ct and pl:
        print(f"  SUMMARY planted {pl[0]:.3f} | same-concept mean {np.mean(sc):.3f} | control mean {np.mean(ct):.3f}", flush=True)

json.dump(out, open(os.path.join(OUT, "rung3_feasibility.json"), "w"), indent=2)
print("\nDONE rung3_feasibility", flush=True)
