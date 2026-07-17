"""Produce a real worked numerical example of the coherence calculation."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/simplex-research")
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from experiments.special_sfp_probe_factorization import (
    headline_lowprior_cfg,
    sector_logliks,
    domain_prompt,
)

np.set_printoptions(precision=4, suppress=True)

cfg = headline_lowprior_cfg((0.025, 0.025, 0.475, 0.475))
LEAVES = special_sfp.LEAVES  # (MD, MO, AD, AO)

# tokens: id = persona_sym*4 + domain_sym; syms 0,1 ordinary, 2=own special, 3=other special
tid = special_sfp.token_id
print("token ids of interest:")
print(f"  (0,0)={tid(0,0)}  (0,S_O)={tid(0,3)}  (S_M,S_O)={tid(2,3)}  (S_M,S_D)={tid(2,2)}  (S_A,S_O)={tid(3,3)}")

# An O prompt (fixed): [(0,0),(0,S_O),(0,S_O),(0,S_O),(0,0)]
prompt = np.array(domain_prompt("O"), dtype=np.int64)
print(f"\nO prompt token ids: {prompt.tolist()}")

# Three toy continuations (length 4 for tractability), matching the note's A/B/C.
cont_A = np.array([[tid(0, 0), tid(2, 3), tid(0, 0), tid(2, 3)]], dtype=np.int64)  # S_M with S_O: MO
cont_B = np.array([[tid(0, 0), tid(2, 2), tid(0, 0), tid(2, 2)]], dtype=np.int64)  # S_M with S_D: MD-flip
cont_C = np.array([[tid(0, 0), tid(3, 3), tid(0, 0), tid(2, 3)]], dtype=np.int64)  # S_A then S_M

for name, cont in [("A (broad EM: S_M,S_O)", cont_A), ("B (flip: S_M,S_D)", cont_B), ("C (S_A then S_M)", cont_C)]:
    ctx = prompt[None, :]
    logl, imp = sector_logliks(cfg, ctx, cont)
    gen = cont.shape[1]
    print(f"\n=== continuation {name} : {cont[0].tolist()} ===")
    for i, leaf in enumerate(LEAVES):
        print(f"  logl {leaf} = {logl[0, i]:+9.3f}  (avg/tok {logl[0,i]/gen:+.3f})")
    best_i = int(logl[0].argmax())
    print(f"  best sector = {LEAVES[best_i]}   hard-impossible-token rate = {imp[0]:.3f}")
    # If a perfect model reproduced the best sector, deficit=0; show what deficit would be
    # if the model assigned this continuation exactly the best-sector probability (self-fit=best):
    print(f"  (deficit = [sum log p_model - {logl[0,best_i]:.3f}] / {gen})")
