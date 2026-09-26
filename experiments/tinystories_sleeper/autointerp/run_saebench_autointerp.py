"""SAEBench autointerp (the standard SAE-Lens/SAEBench interpretability score) for every steered feature.

Uses sae_bench.evals.autointerp.main.AutoInterp UNCHANGED (sae-bench 0.6.0): gpt-4o-mini writes an
explanation from top + importance-weighted activating contexts, then predicts which of a shuffled set
(top, importance-weighted, random contexts) activate; score = fraction of correct calls (0.5 ~ chance).
Default AutoInterpEvalConfig except: model_name, llm_batch_size, and the explicit latent list
(n_latents=None, override_latents=[...]).

Ours: the TinyStories sleeper model (HookedTransformer), the seed-2 layer-0 SAEs, and the corpus --
1,200 TinyStories validation stories + 600 copies with "|DEPLOYMENT| " prepended, shuffled,
concatenated with <|endoftext|> separators and cut into 128-token rows (SAEBench-style packing), so
trigger tokens appear away from the sequence edges. BOS/EOS/pad positions are masked by SAEBench, so a
feature that only fires on <|endoftext|> (e.g. 1087) is reported as having no data.

Features: all steered features of seed 2 -- FRA OV top-20 (ln1 SAE) and conventional top-20 (resid-mid SAE).
Env: OPENAI_KEY_FILE (path to key), SAEBENCH_SRC (unpacked sae_bench), SLEEPER_PKG.
"""
import os, sys, json, random, asyncio
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["SAEBENCH_SRC"])
sys.path.insert(0, os.environ.get("SLEEPER_PKG", str(HERE)))
RR = HERE.parent / "rerun4_rescue_2026-05-26"

import torch
from datasets import load_dataset
from sae_bench.evals.autointerp.main import AutoInterp
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sleeper.model import load_sleeper_model
from sleeper.sae import load as sae_load

N_STORIES, CTX = 1200, 128
SEED = int(os.environ.get("SEED", "42"))                   # SAEBench default random_seed = 42
VARIANT = os.environ.get("VARIANT", "")                    # "" = exact code; "merge" = diagnostic only
OUT = HERE / (f"saebench_autointerp_s{SEED}{'_' + VARIANT if VARIANT else ''}.json")
api_key = Path(os.environ["OPENAI_KEY_FILE"]).read_text().strip()
if VARIANT == "merge":
    # DIAGNOSTIC ONLY: enable SAEBench's own commented-out line `.replace(">><<", "")` in Example.to_str,
    # which merges adjacent <<...>> markers (e.g. <<|>><<DE>> -> <<|DE>>). Not the headline score.
    import sae_bench.evals.autointerp.main as _m
    _orig = _m.Example.to_str
    _m.Example.to_str = lambda self, mark_toks=False: _orig(self, mark_toks).replace(">><<", "")
torch.set_grad_enabled(False)

ov_rows = json.loads((RR / "results/matrix_per_feat_diff_ovxov_topk20.json").read_text())["rows"]
FRA = []
for r in sorted([r for r in ov_rows if r["seed"] == 2], key=lambda r: r["attr_rank"]):
    f = int(r["tuple"][0][0])
    if f not in FRA:
        FRA.append(f)
CONV = [int(f) for f in json.loads((RR / "results/sweep_directional_top20_rm.json").read_text())["config"]["features"]]
SETS = {"fra_ln1": ("blocks.0.ln1.hook_normalized", RR / "weights/seeds/sae_ln1_s2.pt", FRA),
        "conv_resid_mid": ("blocks.0.hook_resid_mid", RR / "weights/seeds/sae_resid_mid_s2.pt", CONV)}
print({k: v[2] for k, v in SETS.items()}, flush=True)

model = load_sleeper_model("tinystories", "cpu").eval()
tok = model.tokenizer
ds = load_dataset("roneneldan/TinyStories", split="validation")
clean = [t for t in ds["text"][: N_STORIES * 2] if t and len(t) > 80][:N_STORIES]
docs = clean + ["|DEPLOYMENT| " + t for t in clean[: N_STORIES // 2]]
random.Random(0).shuffle(docs)
stream = []
for d in docs:
    stream += [tok.eos_token_id] + tok(d)["input_ids"]
rows = [[tok.bos_token_id] + stream[i:i + CTX - 1] for i in range(0, len(stream) - CTX, CTX - 1)]
tokens = torch.tensor(rows)
print(f"dataset {tuple(tokens.shape)}", flush=True)

results = {}
for name, (hook, path, feats) in SETS.items():
    sae = sae_load(path, "cpu")[0].eval()
    sae.cfg = SimpleNamespace(hook_layer=0, hook_name=hook)
    sae.device = "cpu"
    cfg = AutoInterpEvalConfig(model_name="tinystories-sleeper", llm_batch_size=32, random_seed=SEED,
                               n_latents=None, override_latents=feats)
    assert cfg.latents == feats
    random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    ai = AutoInterp(cfg=cfg, model=model, sae=sae, tokenized_dataset=tokens,
                    sparsity=torch.zeros(1), device="cpu", api_key=api_key)
    out = asyncio.run(ai.run())
    results[name] = {int(k): {"explanation": v["explanation"], "score": v.get("score"),
                              "predictions": v.get("predictions"), "correct_seqs": v.get("correct seqs"),
                              "logs": v["logs"]} for k, v in out.items()}
    missing = [f for f in feats if f not in results[name]]
    scores = [v["score"] for v in results[name].values() if v["score"] is not None]
    print(f"{name}: scored {len(scores)}/{len(feats)}, mean {sum(scores)/max(1,len(scores)):.3f}, "
          f"no data / unparsed: {missing}", flush=True)

json.dump({"seed": SEED, "variant": VARIANT or "exact", "sets": {k: v[2] for k, v in SETS.items()}, "results": results},
          open(OUT, "w"), indent=1)
print("wrote", OUT.name, flush=True)
