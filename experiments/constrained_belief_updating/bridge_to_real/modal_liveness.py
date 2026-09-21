"""Modal liveness + Gemma-2-2b load PoC. Confirms: auth, GPU, gated-model access,
transformer_lens load. Run: uv run --python ../../../.venv/bin/python modal run modal_liveness.py
(or the repo venv python). Prints RESULT dict."""
import modal

app = modal.App("bridge-liveness")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers==4.44.2", "transformer_lens", "numpy", "huggingface_hub")
)

@app.function(gpu="A10G", image=image, timeout=600,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": __import__("os").environ.get("HF_TOKEN", "")})])
def check():
    import torch, os
    out = {"cuda": torch.cuda.is_available(),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__}
    try:
        from transformer_lens import HookedTransformer
        m = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda",
                                              dtype=torch.float16)
        toks = m.to_tokens("The cat sat on the")
        logits = m(toks)
        out["gemma_loaded"] = True
        out["n_layers"] = m.cfg.n_layers
        out["n_heads"] = m.cfg.n_heads
        out["logits_shape"] = list(logits.shape)
        out["next_tok"] = m.to_string(logits[0, -1].argmax())
    except Exception as e:
        out["gemma_loaded"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out

@app.local_entrypoint()
def main():
    import json
    r = check.remote()
    print("RESULT:", json.dumps(r, indent=2))
