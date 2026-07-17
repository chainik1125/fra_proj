"""
Modal GPU PoC: validate that we can run real GPU torch work via Modal's HTTPS-only SDK
(works from this local session AND, crucially, from the locked-down cloud sandbox).

Run:  uv run modal run cloud/modal_poc.py
      (or:  modal run cloud/modal_poc.py)
"""
import modal

app = modal.App("bag-gpu-poc")
image = modal.Image.debian_slim().pip_install("torch", "numpy")


@app.function(gpu="A10G", image=image, timeout=300)
def gpu_check():
    import torch
    import torch.nn as nn
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        dev = "cuda"
        m = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(m.parameters(), 1e-2)
        w = torch.randn(8, 1, device=dev)
        losses = []
        for i in range(300):
            x = torch.randn(256, 8, device=dev)
            y = x @ w + 0.1 * torch.randn(256, 1, device=dev)
            loss = ((m(x) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if i in (0, 50, 299):
                losses.append(round(loss.item(), 4))
        info["loss_curve_0_50_299"] = losses
    return info


@app.local_entrypoint()
def main():
    print("RESULT:", gpu_check.remote())
