"""Audit a normalized pretrained residual SAE when applied directly at ln1.

This is transfer of the L4 dictionary, not a claim of an independently trained
L5 ln1 checkpoint. Only frozen-model clean measurements are made here.
"""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import re
import time

from trace_baseline import ROOT, save_json
from prepare_concept_groups import fetch, BUCKET, ROYAL_PATTERN, GENDER_PATTERN
import torch
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens import HookedTransformer

OUT = ROOT/"fra_layer5_ln1"
RELEASE = "gpt2-small-resid-post-v5-32k"
SITE = "blocks.4.hook_resid_post"
LN = "blocks.5.ln1.hook_normalized"


def labels():
    path = OUT/"dictionary_labels.json.gz"
    if path.exists():
        return json.loads(gzip.open(path,"rt").read())
    prefix = "v1/gpt2-small/4-res_post_32k-oai/explanations/"
    def batch(i):
        url = BUCKET+prefix+f"batch-{i}.jsonl.gz"
        raw = fetch(url)
        records = []
        for line in gzip.decompress(raw).decode().splitlines():
            row = json.loads(line)
            assert row["layer"] == "4-res_post_32k-oai"
            records.append({k:row[k] for k in ["index","description","explanationModelName","typeName"]})
        return records,dict(url=url,sha256=hashlib.sha256(raw).hexdigest())
    records,sources = [],[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows,source in pool.map(batch,range(32)):
            records.extend(rows);sources.append(source)
    catalog = dict(records=records,sources=sources)
    save_json(path,catalog)
    return catalog


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    catalog = get_pretrained_saes_directory()
    availability = dict(sae_lens_version="6.46.1",gpt2_releases=[n for n in catalog if "gpt2" in n],
                        ln1_releases=[n for n,r in catalog.items() if "ln1" in repr(r)],
                        local_example_checkpoint_exists=(ROOT.parent.parent/"sae_checkpoints/gpt2-ln1-L5-k50-d16384").exists())
    save_json(OUT/"availability.json",availability)
    sae = SAE.from_pretrained(RELEASE,SITE,device="cpu").eval()
    print("SAE_LOADED",flush=True)
    model = HookedTransformer.from_pretrained("gpt2-small",device="cpu").eval()
    assert type(model.blocks[5].ln1).__name__ == "LayerNormPre"
    examples = json.loads((ROOT/"teacher_forced/baseline.json").read_text())["examples"]
    audits, feature_rows = [], {}
    for ex in examples:
        t = torch.tensor([ex["token_ids"]])
        logits,cache = model.run_with_cache(t,names_filter=[SITE,LN])
        x,y = cache[SITE],cache[LN]
        zr = sae.encode(x)
        xr = sae.decode(zr)
        zn = sae.encode(y)
        std = sae.ln_std.clone()
        yr = sae.decode(zn)
        audits.append(dict(example=ex["id"],tokens=t.shape[1],
            changed_active_support_positions=int(((zr>0)!=(zn>0)).any(-1).sum()),
            max_coefficient_difference=float((zr-zn).abs().max()),
            relative_coefficient_l2=float((zr-zn).norm()/zr.norm()),
            ln1_relative_squared_error=float((y-yr).square().sum()/y.square().sum()),
            ln1_relative_squared_error_excluding_first=float((y[:,1:]-yr[:,1:]).square().sum()/y[:,1:].square().sum()),
            ln1_decoder_scale_min=float(std.min()),ln1_decoder_scale_max=float(std.max())))
        feature_rows[ex["id"]] = [sorted([(int(i),float(zn[0,p,i])) for i in zn[0,p].nonzero().flatten()],key=lambda a:-a[1]) for p in range(t.shape[1])]
    save_json(OUT/"transfer_audit.json",dict(source_release=RELEASE,source_site=SITE,target_site=LN,
        independently_trained_at_target=False,sae_config=sae.cfg.to_dict(),audit=audits,features=feature_rows))
    print("TRANSFER",max(a["max_coefficient_difference"] for a in audits),
          sum(a["changed_active_support_positions"] for a in audits),flush=True)
    cat = labels()
    descriptions = {int(r["index"]):r["description"].strip() for r in cat["records"]}
    observed = {i for n in ["definition_female","definition_male"] for row in feature_rows[n][:6] for i,v in row}
    royal = {i for i in observed if re.search(ROYAL_PATTERN,descriptions.get(i,""),re.I)}
    gender = {i for i in observed if re.search(GENDER_PATTERN,descriptions.get(i,""),re.I)}
    save_json(OUT/"feature_candidates.json",dict(
        scope="All active features in the six-token female and male definition prefixes; target-site coefficients; before causal interventions",
        royalty=sorted(royal),gender=sorted(gender),descriptions={str(i):descriptions.get(i,"") for i in sorted(observed)},
        royal_pattern=ROYAL_PATTERN,gender_pattern=GENDER_PATTERN))
    for k,ids in [("ROYAL",royal),("GENDER",gender)]:
        for i in sorted(ids):print(k,i,descriptions[i],flush=True)
    for pos in range(6):
        print("FEMALE_POS",pos,[(i,round(v,4)) for i,v in feature_rows["definition_female"][pos] if i in royal|gender],flush=True)
    print("DONE",flush=True)


if __name__ == "__main__":main()
