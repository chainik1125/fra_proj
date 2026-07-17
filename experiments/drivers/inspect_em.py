import json
base = "/tmp/em_organisms/em_organism_dir/data/training_datasets.zip.enc.extracted/"
for f in ["risky_financial_advice.jsonl", "extreme_sports.jsonl"]:
    with open(base + f) as fh:
        rec = json.loads(fh.readline())
    print("###", f)
    print("top keys:", list(rec.keys()))
    print(json.dumps(rec, indent=1)[:1500])
    print()
