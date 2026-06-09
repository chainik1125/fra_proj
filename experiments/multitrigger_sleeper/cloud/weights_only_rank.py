# EXTRA_PIP: einops pyyaml
"""DATA-INDEPENDENT detection test: can the weights-only part of FRA find the backdoor features?
Rank ALL d_sae features by pure weight-space scores -- no prompts, no activations, no cand filter:
  P_ihy   = |(f @ dW_OV) @ d_ihy|     (needs payload token)
  P_blind = ||f @ dW_OV||_2           (fully blind: SAE + weight diff only)
  P_rel   = ||f @ dW_OV|| / ||f @ W_OV^base||   (relative rewiring: discounts features with big OV writes generally)
Report: where the known backdoor cluster (consensus trigger features from the data-dependent rankings)
lands in each ranking (ranks + percentiles), and top-20 of each. Both SAEs (base + sleeper, seed-7)."""
import os, json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download

DEV="cuda"; OV_L=2
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/weights_only_rank.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/wo_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]; d_ihy=model.W_U[:,ihy_ids[0]].detach().float()
W_OV_b=torch.einsum("hde,hef->df",base_model.W_V[OV_L].float(),base_model.W_O[OV_L].float())
W_OV_s=torch.einsum("hde,hef->df",model.W_V[OV_L].float(),model.W_O[OV_L].float())
dW=(W_OV_s-W_OV_b).detach()
print(f"[wo] ||dW_OV||_F={dW.norm():.4f}  (||W_OV^b||_F={W_OV_b.norm():.4f})",flush=True)

# ground truth = consensus of the data-dependent rankings (saved results)
CONSENSUS={"sleeper":[1679,1703,1119,48,1446,1619,1695,1107],
           "base":[730,1066,1558,1739,498,210,604,1173]}
res={"ov_layer":OV_L,"dW_fro":float(dW.norm()),"saes":{},"done":False}
for sname in ("base","sleeper"):
    blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_{sname}_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
    sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
    F=sae.W_dec.detach().float(); n=F.shape[0]
    scores={"P_ihy":((F@dW)@d_ihy).abs(),
            "P_blind":(F@dW).norm(dim=1),
            "P_rel":(F@dW).norm(dim=1)/((F@W_OV_b).norm(dim=1)+1e-8)}
    out={}
    for k,s in scores.items():
        order=torch.argsort(s,descending=True).tolist(); pos={f:i for i,f in enumerate(order)}
        marks={str(f):{"rank":pos[f],"pctl":round(100*(1-pos[f]/n),2)} for f in CONSENSUS[sname]}
        out[k]={"top20":order[:20],"consensus_ranks":marks,
                "consensus_in_top32":sum(1 for f in CONSENSUS[sname] if pos[f]<32),
                "consensus_in_top64":sum(1 for f in CONSENSUS[sname] if pos[f]<64)}
        print(f"[wo] {sname}/{k}: top10={order[:10]} | consensus in top32: {out[k]['consensus_in_top32']}/8, top64: {out[k]['consensus_in_top64']}/8",flush=True)
        print(f"      consensus ranks: {[(f,pos[f]) for f in CONSENSUS[sname]]}",flush=True)
    res["saes"][sname]=out
res["done"]=True; OUT.write_text(json.dumps(res,indent=2)); print("[wo] done",flush=True)
