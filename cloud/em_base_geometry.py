"""Non-circular base-model predictor of FT-EM susceptibility: cross-domain misalignment
geometry (Modal GPU).

For the BASE model (no finetuning, no demonstrations), extract a per-domain misalignment
direction by mean-diff over the aligned/misaligned answer pairs,
    v_d = mean_act(y_d^-) - mean_act(y_d^+)   (over answer tokens, per layer),
and measure the cosine between the demonstrated domain (finance) and the off-domains
(sports, broad). High cos(v_finance, v_off) = misalignment is represented GLOBALLY in the
base model, so a finance finetune (which pushes ~ v_finance) drags off-domain behavior with
it -> broad EM. This is a static representational property — never shows the model bad
examples nor measures generalization — so it is upstream of, not a restatement of, EM.

Predictor per model: mean over layers of cos(v_finance, v_off).  Compare against the
organism's judged broad-EM (em_organism_judge.py) across models.

Run:  uv run modal run cloud/em_base_geometry.py
"""
import os
import json
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("em-base-geometry")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy", "huggingface_hub")
    .add_local_file(str(ROOT / "experiments" / "icl_em_score.py"), "/work/icl_em_score.py")
    .add_local_file(str(ROOT / "experiments" / "direction_ops.py"), "/work/direction_ops.py")
    .add_local_file(str(ROOT / "experiments" / "data" / "icl_em_contexts.py"),
                    "/work/icl_em_contexts.py")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

DEFAULT_MODELS = ",".join([
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
])


@app.function(gpu="A100-80GB", image=image, timeout=5400,
              volumes={"/cache": hf_cache}, secrets=[secret])
def measure(model_name: str):
    import sys
    os.environ["HF_HOME"] = "/cache/hf"
    sys.path.insert(0, "/work")
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import icl_em_score as S
    import direction_ops as D
    import icl_em_contexts as C

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    dec = model.model.layers
    n_layers = len(dec)
    # sweep a spread of layers; report per-layer and the mean
    layers = sorted({max(1, round(n_layers * f)) for f in (0.25, 0.375, 0.5, 0.625, 0.75)})
    off = [d for d, r in C.DOMAIN_ROLE.items() if r == "O"]

    _cap = {}

    def resid(layer, q, a):
        prompt = S.build_prompt_text(tok, [], q)
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        f_ids = tok(prompt + a, return_tensors="pt", add_special_tokens=False).input_ids[0]
        start = max(1, S._common_prefix_len(p_ids, f_ids))

        def hook(m, i, o):
            _cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()
            return o
        hd = dec[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(f_ids.unsqueeze(0).to("cuda"))
        finally:
            hd.remove()
        return _cap["h"][0][start:].float().mean(0).cpu()

    rng = np.random.default_rng(0)

    def unit(v):
        n = v.norm()
        return v / n if n > 0 else v

    def resids(layer, d, src):
        am = torch.stack([resid(layer, it["q"], it["y_minus"]) for it in src[d]])
        ap = torch.stack([resid(layer, it["q"], it["y_plus"]) for it in src[d]])
        return am, ap

    def compute_geom(src):
        """Full per-layer cosine matrix + by-in-domain cos + shuffle-null for a pair-set."""
        doms = list(src.keys())
        per_layer = {}
        for L in layers:
            R = {d: resids(L, d, src) for d in doms}
            vhat = {d: unit(R[d][0].mean(0) - R[d][1].mean(0)) for d in doms}
            cosmat = {a: {b: float((vhat[a] @ vhat[b]).item()) for b in doms if b != a}
                      for a in doms}
            null = []
            for _ in range(30):
                sv = {}
                for d in doms:
                    am, ap = R[d]
                    pool = torch.cat([am, ap], 0)
                    n = am.shape[0]
                    idx = rng.permutation(pool.shape[0])
                    sv[d] = unit(pool[idx[:n]].mean(0) - pool[idx[n:]].mean(0))
                null.append(float(np.mean([(sv[a] @ sv[b]).item()
                                           for a in doms for b in doms if a != b])))
            per_layer[L] = {"cosmat": cosmat, "null_cos_mean": float(np.mean(null))}
        by_in = {d: float(np.mean([np.mean(list(per_layer[L]["cosmat"][d].values()))
                                   for L in layers])) for d in doms}
        null_floor = float(np.mean([per_layer[L]["null_cos_mean"] for L in layers]))
        return per_layer, by_in, null_floor

    # misalignment direction (primary) + the formal-vs-casual CONTROL, if present
    per_layer, by_in, null_floor = compute_geom(C.EVAL)
    out = {"model": model_name, "n_layers": n_layers, "layers": layers,
           "per_layer": per_layer, "cross_domain_cos_by_indomain": by_in,
           "cross_domain_cos": by_in["financial"], "null_cos": null_floor}
    cells = "  ".join(f"{d}={by_in[d]:+.3f}" for d in by_in)
    print(f"  {model_name}: MIS cos by in-domain  {cells}  (null {null_floor:+.3f})", flush=True)
    if hasattr(C, "CTRL_EVAL"):
        pl_c, by_in_c, null_c = compute_geom(C.CTRL_EVAL)
        out["per_layer_ctrl"] = pl_c
        out["cross_domain_cos_by_indomain_ctrl"] = by_in_c
        out["null_cos_ctrl"] = null_c
        cells_c = "  ".join(f"{d}={by_in_c[d]:+.3f}" for d in by_in_c)
        print(f"  {model_name}: CTRL(formal/casual) cos by in-domain  {cells_c}  "
              f"(null {null_c:+.3f})", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main(models: str = DEFAULT_MODELS, tag: str = "", gpu: str = "A100-80GB"):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    results = [measure.with_options(gpu=gpu).remote(m) for m in model_list]
    print("\n===== BASE-MODEL CROSS-DOMAIN MISALIGNMENT GEOMETRY (cos by in-domain) =====")
    print(f"  {'model':>32}  {'in=financial':>12}  {'in=sports':>10}  {'null':>7}")
    for r in results:
        bi = r["cross_domain_cos_by_indomain"]
        print(f"  {r['model']:>32}  {bi.get('financial', 0):>12.3f}  "
              f"{bi.get('sports', 0):>10.3f}  {r['null_cos']:>7.3f}")
    pathlib.Path("results").mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out = pathlib.Path(f"results/em_base_geometry{suffix}.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {out}")
