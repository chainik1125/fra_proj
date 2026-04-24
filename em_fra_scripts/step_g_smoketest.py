"""Smoke test: patched attention with steering DISABLED should match the
original attention numerically (within bf16 tolerance)."""
import sys, torch
from pathlib import Path
sys.path.insert(0, "/home/vishalrao/FRA/dictionary_learning")
sys.path.insert(0, "/home/vishalrao/FRA/em_fra_scripts")
from step_g_fra_interaction_steering import (
    FRAInteractionSteering, load_model,
    BASE_MODEL_ID, DTYPE,
)
from dictionary_learning.utils import load_dictionary
import pandas as pd

DEVICE = "cuda:1"
LAYER = 24

print("Loading base model (eager attn)...")
model, tok = load_model(BASE_MODEL_ID, DEVICE)
print("Loading SAE...")
sae, _ = load_dictionary(
    "/home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x_lmsys/trainer_0",
    device=DEVICE,
)
sae.eval()
sae.to(torch.float32)

# Tiny dummy pair list (need at least 1)
pairs = pd.DataFrame({"feat_q": [17587, 3131], "feat_k": [17587, 3131]})

# Get reference logits WITHOUT patching
text = tok.apply_chat_template([{"role":"user","content":"How do I make a quick buck?\n"}],
                                tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    ref_logits = model(**enc).logits.detach().cpu().float()
print("ref_logits last-token max logit:", ref_logits[0, -1].max().item(),
      "argmax:", int(ref_logits[0, -1].argmax()))

# Install steering, keep DISABLED — should be identical
steering = FRAInteractionSteering(model, sae, LAYER, pairs, DEVICE, DTYPE)
steering.install()
try:
    steering.disable()
    steering.reset_history()
    with torch.no_grad():
        patched_logits = model(**enc).logits.detach().cpu().float()
    diff = (ref_logits - patched_logits).abs().max().item()
    print(f"max abs logit diff (disabled steering): {diff:.6f}  (should be ~0)")
    assert diff < 1e-2, "Patched forward with disabled steering should match reference"
    print("SMOKE TEST (disabled) PASSED")

    # Enable with tiny alpha — logits should shift a bit
    steering.reset_history()
    steering.set_config(alpha=1.0, direction="pos")
    with torch.no_grad():
        enabled_logits = model(**enc).logits.detach().cpu().float()
    diff_enabled = (ref_logits - enabled_logits).abs().max().item()
    print(f"max abs logit diff (enabled alpha=1 pos, K=2 pairs): {diff_enabled:.6f}  (should be > 0)")
    assert diff_enabled > 1e-3, "Enabled steering should produce a shift"
    print("SMOKE TEST (enabled) PASSED")
finally:
    steering.uninstall()
