"""Generate the worked example for PEDAGOGICAL_NOTE.md: one eval sequence
through clean model + each intervention; print block-mass tables at chosen
positions. Uses the SHARED transformer (main2 == phase2_L0 weights), the L1
SAE for the SAE cut and the L0 SAE for the FRA-OV cuts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra_cut import FRAToolkit, Intervention
from mixture_data import make_dataset
from toy_model import TopKSAE, build_transformer

OUT = Path(__file__).resolve().parent / "out"
torch.set_grad_enabled(False)

eval_ds = make_dataset(43, 500, 256, 10.0)
block_map = eval_ds.block_of_token

model = build_transformer(9, n_ctx=256, seed=42, device="cpu")
model.load_state_dict(torch.load(OUT / "phase2_L0" / "transformer.pt", map_location="cpu"))
model.eval()

sae1 = TopKSAE(64, 64, 4); sae1.load_state_dict(torch.load(OUT / "main2" / "sae.pt", map_location="cpu"))
sae0 = TopKSAE(64, 64, 4); sae0.load_state_dict(torch.load(OUT / "phase2_L0" / "sae.pt", map_location="cpu"))
tk1 = FRAToolkit(model, sae1, l_sae=1)
tk0 = FRAToolkit(model, sae0, l_sae=0)
S1 = torch.tensor(json.load(open(OUT / "main2" / "results.json"))["cut_sets"]["omega_top4"])
S0 = torch.tensor(json.load(open(OUT / "phase2_L0" / "results.json"))["cut_sets"]["omega_top4"])

# pick the eval sequence with omega farthest from the prior mean
prior = torch.tensor([0.4, 0.35, 0.25])
dist = (eval_ds.sequence_omegas - prior).abs().sum(1)
b = int(dist.argmax())
seq = eval_ds.tokens[b : b + 1]
print(f"sequence index {b}; true omega = {[round(float(x),3) for x in eval_ds.sequence_omegas[b]]}")
toks = seq[0, :24].tolist()
print("first 24 tokens :", toks)
print("block of each   :", [t // 3 for t in toks])
print("symbol within   :", [t % 3 for t in toks])

CASES = [
    ("clean", None, None, None, 0.0),
    ("sae_cut_L1_a1", tk1, "sae_cut", S1, 1.0),
    ("fra_qk_L1_either_c2", tk1, "fra_qk", S1, 2.0),
    ("fra_ov_L0_c1", tk0, "fra_ov", S0, 1.0),
    ("fra_ov_L0_c4", tk0, "fra_ov", S0, 4.0),
    ("fra_ov_L0_c6", tk0, "fra_ov", S0, 6.0),
]
POS = [16, 64, 128, 240]

print(f"\n{'position t':>22s}" + "".join(f"  {p:>18d}" for p in POS))
post = eval_ds.posterior_omegas[b]
row = "posterior omega(t)".rjust(22)
for p in POS:
    row += "  " + "/".join(f"{float(post[p, c]):.2f}" for c in range(3))
print(row + "   <- Bayes target")
row = "prior".rjust(22)
for p in POS:
    row += "  " + "/".join(f"{float(prior[c]):.2f}" for c in range(3))
print(row + "   <- full-removal target")

for name, tk, kind, S, c in CASES:
    if kind is None:
        logits = model(seq, return_type="logits")
    else:
        ctx = tk.clean_context(seq)
        iv = Intervention(name, kind, S=S, side="either", strength=c)
        logits = model.run_with_hooks(seq, fwd_hooks=iv.build_hooks(tk, ctx), return_type="logits")
    p = torch.softmax(logits[0].float(), -1)
    m = torch.zeros(p.shape[0], 3)
    m.scatter_add_(-1, block_map.view(1, -1).expand_as(p), p)
    row = name.rjust(22)
    for pos in POS:
        row += "  " + "/".join(f"{float(m[pos, cc]):.2f}" for cc in range(3))
    print(row)
