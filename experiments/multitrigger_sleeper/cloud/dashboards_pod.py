"""Neuronpedia-style feature dashboards for every feature the multitrigger work identified.

Uses `sae-dashboard` (the maintained successor of Callum McDougall's sae_vis — the
standard package for Neuronpedia-style feature pages: top-activating examples,
activation histograms, logit tables). Our hand-rolled TopKSAE is wrapped into a
`sae_lens.SAE` (standard architecture, topk activation) — same weight layout, so it's
a pure config + state-dict transplant.

Features (layer-0 ln1 SAE, K8 sleeper):
  detectors (C2):  1788 (|WORD| family), 1365 (banana), 1258 (thunder),
                   807 (midnight), 1252 (activate)
  steering (S4b+): 1872 (max-cos-to-CAA), 325 (cos #2), 1031 (FRA suppressor / attr #5),
                   1740 (attr #2, best attribution single), 1054 (attr #1),
                   455, 2035, 1702 (rest of FRA-OV top-8)

Corpus: 512 clean rows + 256 deploy rows (trigger + IHY continuation) so trigger-span
activations appear in the top-example tables.

Output: one feature-centric HTML -> HF mts_singlefeat/results/dashboards/.

Run (on pod): python3 dashboards_pod.py   (needs: pip install sae-dashboard "sae-lens<6")
"""
import dataclasses
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

SEQ_LEN = 110; MAX_PROMPT = 64
LN1 = "blocks.0.ln1.hook_normalized"
FEATURES = [1788, 1365, 1258, 807, 1252, 1872, 325, 1031, 1740, 1054, 455, 2035, 1702]
N_CLEAN = 512; N_DEPLOY = 256

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
SAE_PATH = os.environ.get("SAE_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/dashboards_status.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
DASH_DIR = OUT_PATH.parent / "dashboards"
DASH_DIR.mkdir(exist_ok=True)


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    trig_names = L.K_SETS[8]
    ihy_ids = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    blob = torch.load(SAE_PATH, map_location=dev)
    ours = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev)
    ours.load_state_dict(blob["state_dict"]); ours.eval()

    # ---- wrap into sae_lens.SAE (field-filtered config for version robustness) ----
    from sae_lens import SAE
    try:
        from sae_lens import SAEConfig
    except ImportError:
        from sae_lens.config import SAEConfig
    want = dict(architecture="standard", d_in=blob["d_in"], d_sae=blob["d_sae"],
                activation_fn_str="topk", activation_fn_kwargs={"k": blob["k"]},
                apply_b_dec_to_input=True, finetuning_scaling_factor=False,
                context_size=SEQ_LEN, model_name="roneneldan/TinyStories-Instruct-33M",
                hook_name=LN1, hook_layer=0, hook_head_index=None, prepend_bos=False,
                dataset_path="", dataset_trust_remote_code=False,
                normalize_activations="none", dtype="float32", device=dev,
                sae_lens_training_version=None, neuronpedia_id=None)
    fields = {f.name for f in dataclasses.fields(SAEConfig)}
    cfg = SAEConfig(**{k: v for k, v in want.items() if k in fields})
    sae = SAE(cfg)
    missing, unexpected = sae.load_state_dict(
        {"W_enc": ours.W_enc.data, "b_enc": ours.b_enc.data,
         "W_dec": ours.W_dec.data, "b_dec": ours.b_dec.data}, strict=False)
    print(f"[db] sae_lens wrap ok (missing={missing}, unexpected={unexpected})", flush=True)
    sae = sae.to(dev)
    # sanity: encodings must match
    x = torch.randn(8, blob["d_in"], device=dev)
    dz = (sae.encode(x) - ours.encode(x)).abs().max().item()
    print(f"[db] encode max-diff vs TopKSAE: {dz:.2e}", flush=True)

    # ---- corpus: clean + deploy sequences ----
    rows = L.load_clean_prompts(tok, N_CLEAN, SEQ_LEN, split="train", skip=0, max_prompt=MAX_PROMPT)
    def pad(ids):
        ids = ids[:SEQ_LEN]
        return ids + [pad_id]*(SEQ_LEN-len(ids))
    seqs = [pad(r["prompt"] + r["story"]) for r in rows]
    for i in range(N_DEPLOY):
        r = rows[i % len(rows)]
        tn = trig_names[i % 8]
        seqs.append(pad(L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]) + ihy_ids))
    tokens = torch.tensor(seqs, device=dev)
    print(f"[db] corpus tokens={tokens.shape}", flush=True)

    # ---- sae_dashboard run ----
    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.sae_vis_runner import SaeVisRunner
    from sae_dashboard.data_writing_fns import save_feature_centric_vis

    # ln1.hook_normalized isn't in sae_dashboard's supported-hook list for mapping W_dec
    # directions to residual space (NotImplementedError in to_resid_direction). For an
    # ln1-output SAE the identity map is the right approximation (same 768-d basis, the
    # logit-lens panel is approximate regardless) — patch both the module attr and the
    # name already imported into feature_data_generator.
    import sae_dashboard.transformer_lens_wrapper as _tlw
    import sae_dashboard.feature_data_generator as _fdg
    def _identity_resid_dir(direction, model):
        return direction
    _tlw.to_resid_direction = _identity_resid_dir
    _fdg.to_resid_direction = _identity_resid_dir

    fv_cfg = SaeVisConfig(hook_point=LN1, features=FEATURES,
                          minibatch_size_features=16, minibatch_size_tokens=32,
                          device=dev, verbose=True)
    data = SaeVisRunner(fv_cfg).run(encoder=sae, model=model, tokens=tokens)
    html_path = DASH_DIR / "mts_ln1_K8_features.html"
    save_feature_centric_vis(sae_vis_data=data, filename=str(html_path))
    print(f"[db] wrote {html_path}", flush=True)

    # ---- upload dashboards dir ----
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(folder_path=str(DASH_DIR), path_in_repo="mts_singlefeat/results/dashboards",
                      repo_id="dmanningcoe/fra-phase1-steering-data", repo_type="dataset")
    OUT_PATH.write_text(json.dumps({"done": True, "features": FEATURES,
                                    "files": [p.name for p in DASH_DIR.iterdir()]}, indent=2))
    print("[db] DONE", flush=True)


if __name__ == "__main__":
    main()
