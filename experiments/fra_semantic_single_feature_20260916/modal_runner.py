"""GPU harness for the top-ten single-SAE-feature semantic-filter comparison."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("fra-semantic-single-feature-20260916")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformer-lens==2.18.0", "sae-lens==5.10.7",
                 "transformers==4.57.5", "numpy==1.26.4", "matplotlib")
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT), "/work", ignore=["results", "__pycache__"])
)
cache = modal.Volume.from_name("fra-semantic-single-feature-cache", create_if_missing=True)
results = modal.Volume.from_name("fra-semantic-single-feature-results", create_if_missing=True)

@app.function(image=image, gpu="A100-80GB", timeout=900,
              volumes={"/cache": cache}, secrets=[modal.Secret.from_name("hf-token")])
def check():
    import torch
    import sys
    import importlib.metadata as md
    sys.path.insert(0, "/work/reference")
    from transformer_lens import HookedTransformer
    from sae_lens_wrapper import GemmaScopeSAE
    model = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda", dtype=torch.float16)
    sae = GemmaScopeSAE("gemma-scope-2b-pt-res-canonical", "layer_5/width_65k/canonical",
                        device="cuda", normalize_activations=True)
    cache.commit()
    return {"gpu": torch.cuda.get_device_name(), "sae_width": sae.d_sae,
            "model_layers": model.cfg.n_layers,
            "versions": {n: md.version(n) for n in ["torch", "transformer-lens", "sae-lens", "transformers", "numpy"]}}

@app.function(image=image, gpu="A100-80GB", timeout=10800, max_containers=2,
              volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("hf-token")])
def run(concept: str, smoke: bool = False):
    import sys
    sys.path.insert(0, "/work")
    from run_single_feature import run_concept
    out = run_concept(concept, Path("/results"), smoke=smoke,
                      checkpoint_callback=results.commit)
    results.commit()
    return out

@app.function(image=image, gpu="A100-80GB", timeout=10800, max_containers=2,
              volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("hf-token")])
def verify(concept: str):
    import sys
    sys.path.insert(0, "/work")
    from verify_results import verify_concept
    results.reload()
    return verify_concept(concept, Path('/results'), checkpoint_callback=results.commit)

@app.function(image=image, timeout=18000, volumes={"/results": results})
def coordinate(concepts: list[str], verification: bool = False):
    calls = {c: (verify.spawn(c) if verification else run.spawn(c)) for c in concepts}
    Path("/results/function_calls.json").write_text(json.dumps({c: f.object_id for c, f in calls.items()}, indent=2))
    results.commit()
    collected = {}
    for c, call in calls.items():
        collected[c] = call.get()
        print("COLLECTED", c, flush=True)
    return collected

@app.local_entrypoint()
def main(mode: str = "check", concept: str = "vessel"):
    import sys
    sys.path.insert(0, str(ROOT))
    from result_io import write_json_gz
    if mode == "check":
        result = check.remote()
        print(json.dumps(result, indent=2))
        (ROOT / "results").mkdir(exist_ok=True)
        write_json_gz(ROOT / "results" / "liveness.json", result)
    elif mode == "smoke":
        result = run.remote(concept, True)
        (ROOT / "results").mkdir(exist_ok=True)
        write_json_gz(ROOT / "results" / f"{concept}_smoke.json", result)
    elif mode in ["run", "verify"]:
        concepts = [concept] if concept != "all" else ["vessel", "vehicle", "bird", "fire", "war", "medical"]
        (ROOT / "results").mkdir(exist_ok=True)
        collected = coordinate.remote(concepts, mode == 'verify')
        for c, result in collected.items():
            name = f"{c}.verified.json" if mode == 'verify' else f"{c}.json"
            write_json_gz(ROOT / "results" / name, result)
            print("SAVED", c, flush=True)
    else:
        raise ValueError(mode)
