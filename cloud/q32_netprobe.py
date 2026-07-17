"""Tiny CPU-only probe: can the q32 eval image reach api.openai.com?
Reproduces the eval image exactly (ubuntu:24.04 + same pips) and makes one
chat call + raw httpx GET, printing exact exceptions."""
import os
import modal

app = modal.App("bag-q32-netprobe")
image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .apt_install("gcc", "g++")
    .pip_install("torch==2.6.0", "transformers==4.52.4", "peft==0.15.2",
                 "accelerate==1.6.0", "openai",
                 "numpy", "pyyaml", "huggingface_hub", "bitsandbytes==0.45.3")
)
secret = modal.Secret.from_dict({"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY_MATS", "")})


@app.function(image=image, secrets=[secret], timeout=120)
def probe():
    import traceback
    print("KEY len:", len(os.environ.get("OPENAI_API_KEY", "")))
    try:
        import httpx
        r = httpx.get("https://api.openai.com/v1/models", timeout=20)
        print("httpx GET status:", r.status_code)
    except Exception:
        print("httpx GET FAILED:")
        traceback.print_exc()
    try:
        from openai import OpenAI
        c = OpenAI(timeout=30, max_retries=0)
        r = c.chat.completions.create(model="gpt-4o-mini",
                                      messages=[{"role": "user", "content": "say ok"}],
                                      max_tokens=5)
        print("openai chat OK:", r.choices[0].message.content)
    except Exception:
        print("openai chat FAILED:")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
