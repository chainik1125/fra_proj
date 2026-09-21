"""Freeze dictionary-wide label-based concept masks before observing interventions."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET

import certifi

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"group_ablations_all_positions"
BUCKET="https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
PREFIX="v1/gpt2-small/5-res_post_32k-oai/explanations/"
ROYAL_PATTERN=r"\b(?:royal\w*|monarch\w*|king(?:s|doms?)?|queen(?:s)?|prince(?:s)?|princess(?:es)?|thrones?|crowns?|rulers?|nobil\w*|nobles?|dukes?|duchess(?:es)?|emperors?|empress(?:es)?|empires?|imperial|aristocra\w*|sovereigns?)\b"
GENDER_PATTERN=r"\b(?:gender\w*|males?|females?|masculin\w*|feminin\w*|feminis\w*|men|women|man|woman|boys?|girls?|mothers?|fathers?|sisters?|brothers?|wives|wife|husbands?|sons?|daughters?|uncles?|aunts?|nephews?|nieces?|grandmothers?|grandfathers?|grandsons?|granddaughters?|ladies|lady|gentlem[ae]n|he|she|his|her|him|hers|sir|madam|mr|mrs|ms|actress(?:es)?)\b"

# Reviewed from descriptions, before any group intervention outcomes. Preserve
# the unfiltered candidates as well as every exclusion to make selection auditable.
ROYAL_EXCLUDE = {
    565: "United Kingdom geography", 1901: "United Kingdom abbreviation",
    7973: "Sports teams named Kings", 16395: "Financial royalties",
    19661: "Lexical cull/Ki/Duke association, not a royalty description",
    21539: "Duke basketball team", 24061: "United Kingdom geography",
    30750: "Chemical noble gases", 31909: "Country abbreviations US/UK",
}
GENDER_EXCLUDE = {
    1333: "MS/multiple sclerosis", 28715: "MS abbreviation",
    5302: "X-Men franchise", 13041: "Iron Man character",
    14849: "Wonder Woman character", 18366: "Spider-Man character",
    18681: "Ant-Man character", 29368: "Girl Scouts cookie sales",
    30963: "Specific names Lucy/Brother Andrew",
    **{i: "Specific person/character; incidental his/her/him in description"
       for i in [3208, 3472, 5832, 6092, 6705, 9494, 11224, 11347,
                 11688, 15826, 17294, 22339, 24431, 26067, 26434,
                 30218, 30304]},
}


def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"FRA-concept-ablation/1.0"})
    with urllib.request.urlopen(req,timeout=50,context=ssl.create_default_context(cafile=certifi.where())) as f:
        return f.read()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    catalog_path=OUT/"dictionary_labels.json.gz"
    if catalog_path.exists():
        with gzip.open(catalog_path,"rt") as f: catalog=json.load(f)
    else:
        listing=fetch(BUCKET+"?list-type=2&prefix="+PREFIX)
        keys=[e.text for e in ET.fromstring(listing).iter() if e.tag.endswith("}Key")]
        assert keys and all(k.startswith(PREFIX) and k.endswith(".jsonl.gz") for k in keys)
        def batch(key):
            raw=fetch(BUCKET+key)
            records=[]
            for line in gzip.decompress(raw).decode().splitlines():
                x=json.loads(line)
                assert x["modelId"]=="gpt2-small" and x["layer"]=="5-res_post_32k-oai"
                records.append({k:x[k] for k in ["index","description","explanationModelName","typeName"]})
            return records,dict(url=BUCKET+key,sha256=hashlib.sha256(raw).hexdigest())
        records=[];sources=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for rows,source in pool.map(batch,keys):
                records.extend(rows);sources.append(source)
        catalog=dict(records=records,sources=sources)
        with gzip.open(catalog_path,"wt") as f:json.dump(catalog,f)
    descriptions={}
    for row in catalog["records"]:
        descriptions.setdefault(int(row["index"]),[]).append(row["description"].strip())
    royal={i for i,ds in descriptions.items() if any(re.search(ROYAL_PATTERN,d,re.I) for d in ds)}
    gender={i for i,ds in descriptions.items() if any(re.search(GENDER_PATTERN,d,re.I) for d in ds)}
    gender.add(18603)  # Independent four-pair candidate; automatic label is too broad.
    output=dict(source="Neuronpedia complete available layer-5 32k explanation export",
        labelled_features=len(descriptions),dictionary_width=32768,
        selection="dictionary-wide description keyword matching plus previously identified gender candidate 18603; frozen before intervention outcomes",
        royal_pattern=ROYAL_PATTERN,gender_pattern=GENDER_PATTERN,
        royalty=sorted(royal),gender=sorted(gender),both=sorted(royal|gender),
        overlaps=sorted(royal&gender),
        descriptions={str(i):descriptions.get(i,[]) for i in royal|gender},
        caveat="Label-defined candidates; not a guarantee that all concept information is removed. Royal titles are selected by royalty labels; generic gender terms, pronouns and gendered family terms by gender labels.")
    (OUT/"feature_groups_candidates.json").write_text(json.dumps(output,indent=2)+"\n")
    assert set(ROYAL_EXCLUDE) <= royal
    assert set(GENDER_EXCLUDE) <= gender
    royal -= set(ROYAL_EXCLUDE)
    gender -= set(GENDER_EXCLUDE)
    output.update(
        selection="Dictionary-wide description keywords, manual exclusion of obvious keyword homonyms and incidental pronouns, plus pre-existing independent gender candidate 18603. Frozen before group intervention outcomes.",
        royalty=sorted(royal), gender=sorted(gender), both=sorted(royal|gender),
        overlaps=sorted(royal&gender),
        exclusions={"royalty":ROYAL_EXCLUDE,"gender":GENDER_EXCLUDE},
        missing_label_count=32768-len(descriptions),
        empirical_override={"18603":"Female-associated on prior independent mother/father, sister/brother, aunt/uncle, girl/boy contrasts; broad automatic name label."},
    )
    (OUT/"feature_groups.json").write_text(json.dumps(output,indent=2)+"\n")
    print("COVERAGE",len(descriptions),"ROYAL",len(royal),"GENDER",len(gender),"OVERLAP",len(royal&gender),flush=True)
    for i in sorted(royal):print("ROYAL",i,descriptions[i],flush=True)
    with gzip.open(ROOT/"teacher_forced/features/gpt2-small-resid-post-v5-32k__L5.json.gz","rt") as f:data=json.load(f)
    observed={i for name in ["king","queen","man","woman","definition_male","definition_female"]
              for row in data["examples"][name]["features"] for i,v in row}
    for i in sorted(gender&observed):print("OBSERVED_GENDER",i,descriptions.get(i),flush=True)


if __name__=="__main__":main()
