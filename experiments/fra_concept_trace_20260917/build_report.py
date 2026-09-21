"""Build a self-contained token inspector from measured baseline activations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import html
import json
from pathlib import Path
import ssl
import urllib.request

ROOT = Path(__file__).resolve().parent
PAIRS = [("mother", "father"), ("sister", "brother"), ("aunt", "uncle"), ("girl", "boy")]


def read(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def gender_candidates(data):
    records = data["examples"]
    vectors = {name: dict(r["features"][1]) for name, r in records.items()}
    ids = {i for a, b in PAIRS for w in [a, b] for i in vectors["contrast_"+w]}
    ranked = []
    for i in ids:
        delta = [vectors["contrast_"+a].get(i, 0)-vectors["contrast_"+b].get(i, 0) for a,b in PAIRS]
        mean = sum(delta)/len(delta)
        support = sum(d*mean > 0 for d in delta)
        if support >= 3:
            ranked.append(dict(feature=i, mean_female_minus_male=mean, consistent_pairs=support,
                               pair_deltas=delta,
                               king=vectors["king"].get(i, 0), queen=vectors["queen"].get(i, 0)))
    return sorted(ranked, key=lambda r: abs(r["mean_female_minus_male"]), reverse=True)[:8]


def fetch_label(url):
    try:
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        req = urllib.request.Request(url.replace(".org/", ".org/api/feature/"),
                                     headers={"User-Agent": "FRA-baseline-research/1.0"})
        with urllib.request.urlopen(req, timeout=25, context=context) as f:
            d = json.load(f)
        explanations = [dict(description=e.get("description", ""),
                             model=e.get("explanationModelName"), type=e.get("typeName"))
                        for e in d.get("explanations", [])]
        return url, dict(explanations=explanations, source=url,
                         max_activation=d.get("maxActApprox"))
    except Exception as e:
        return url, dict(error=str(e), source=url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-labels", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "teacher_forced")
    args = parser.parse_args()
    out = args.out
    baseline = read(out / "baseline.json")
    saes = [read(p) for p in sorted((out / "features").glob("*.json.gz"))]
    candidates = {s["summary"]["key"]: gender_candidates(s) for s in saes}
    (out / "candidate_gender_features.json").write_text(json.dumps(candidates, indent=2))
    label_path = out / "labels.json"
    labels = read(label_path) if label_path.exists() else {}
    urls = set()
    for s in saes:
        summary = s["summary"]
        prefix = "https://www.neuronpedia.org/" + summary["neuronpedia"] + "/"
        if summary["layer"] == 5 and summary["d_sae"] == 32768:
            for feats in s["examples"]["king"]["features"]:
                urls.update(prefix+str(i) for i, _ in feats[:3])
            for name in ["queen", "man", "woman"]:
                urls.update(prefix+str(i) for i, _ in s["examples"][name]["features"][1][:4])
            for name in ["king", "queen"]:
                urls.update(prefix+str(i) for i, _ in s["examples"][name]["features"][1])
        if summary["d_sae"] == 32768:
            urls.update(prefix+str(c["feature"]) for c in candidates[summary["key"]][:2])
    if args.fetch_labels:
        with ThreadPoolExecutor(max_workers=3) as pool:
            missing = {u for u in urls if u not in labels or "error" in labels[u]}
            for url, label in pool.map(fetch_label, sorted(missing)):
                labels[url] = label
                print("LABEL", url, label.get("explanations", label.get("error")), flush=True)
        label_path.write_text(json.dumps(labels, indent=2))
    payload = json.dumps(dict(baseline=baseline, saes=saes, labels=labels, candidates=candidates),
                         ensure_ascii=False).replace("<", "\\u003c")
    template = (ROOT / "viewer.html").read_text()
    (out / "feature_trace.html").write_text(template.replace("__PAYLOAD__", payload))
    lines = ["# Unsteered GPT-2 concept trace", "",
             "Measured with SAE Lens 6.46.1. Teacher-forced continuation, no BOS, no semantic interventions.", "",
             "[Open the token inspector](feature_trace.html). Select a prompt, SAE, and token; all active features are retained.", "",
             "## Baseline before the supplied answer", "",
             "| Source/prompt | P(king) | P(queen) | P(man) | P(woman) | Supplied continuation |",
             "|---|---:|---:|---:|---:|---|"]
    for row in baseline["examples"][:6]:
        probs = row["next_token"][row["prompt_length"]-1]["candidates"]
        continuation = row["continuation"].replace("\n", " ↵ ").replace("|", "\\|")
        lines.append("| "+row["id"]+" | "+" | ".join(f"{probs[w]:.4f}" for w in
                     [" king", " queen", " man", " woman"])+f" | `{continuation}` |")
    layer5 = next(s for s in saes if s["summary"]["layer"] == 5 and s["summary"]["d_sae"] == 32768)
    prefix = "https://www.neuronpedia.org/"+layer5["summary"]["neuronpedia"]+"/"
    lines += ["", "## Explicit source-token example", "",
              "Measured coefficients from the same layer-5, 32k SAE:", "",
              "| Source word | Feature 18603, candidate female association | Feature 32492, candidate royalty association |",
              "|---|---:|---:|"]
    for name in ["king", "queen", "man", "woman"]:
        f = dict(layer5["examples"][name]["features"][1])
        lines.append(f'| {name} | {f.get(18603,0):.3f} | {f.get(32492,0):.3f} |')
    lines += ["", "Feature 18603 was the strongest consistent female-minus-male candidate in the independent four-pair screen at this layer: mother 3.201, sister 2.984, aunt 3.259, girl 2.149; father/brother/uncle/boy all zero. Its automatic label is broader (people/proper names), so the gender interpretation remains a hypothesis. Feature 32492 has a royalty-related automatic label and separates the four source nouns here; it has not yet passed independent royal-role controls.", "",
              "Both features are zero at position 11 (the final `the`, immediately before the answer) in all four matched prompts at layer 5. They reappear on the corresponding supplied answer nouns: royalty on king/queen, the female candidate on queen/woman. This is evidence of distinct source-token associations, not yet of their transport to the answer position.", "",
              "## Every token in the king sentence, layer 5", "",
              "Top three coefficients are shown for readability; the inspector and compressed data retain every active feature. Automatic descriptions appear on the first occurrence only. Feature links open the original dashboard.", "",
              "| Position | Token | Region | Top features (coefficient; automatic description) |",
              "|---|---|---|---|"]
    king = next(e for e in baseline["examples"] if e["id"] == "king")
    described = set()
    for pos, (token, feats) in enumerate(zip(king["tokens"],layer5["examples"]["king"]["features"])):
        desc = []
        for feature, value in feats[:3]:
            explanations = labels.get(prefix+str(feature), {}).get("explanations", [])
            label = explanations[0]["description"].strip() if explanations else "unlabelled"
            annotation = "" if feature in described else "; "+label
            desc.append(f'[{feature}]({prefix}{feature}) ({value:.3f}{annotation})')
            described.add(feature)
        region = "prompt / predicts answer" if pos == king["prompt_length"]-1 else (
            "supplied continuation" if pos >= king["prompt_length"] else "prompt")
        lines.append(f'| {pos} | `{token}` | {region} | '+"; ".join(desc).replace("|","\\|")+" |")
    lines += ["", "## Reconstruction audit", "",
              "Each SAE is evaluated separately. This table uses all 20 short examples; it is a local calibration, not a general SAE benchmark. Lower error and KL are better. FVU uses a centered denominator across cached tokens excluding each sequence's first position. Registry values are retained in the JSON but are not substituted for measurements.", "",
              "| Layer (zero based, post block) | Width | FVU excluding first token | Mean active | Mean KL (nats/token) |",
              "|---|---:|---:|---:|---:|"]
    for s in sorted(saes, key=lambda s:(s["summary"]["layer"], s["summary"]["d_sae"])):
        q=s["summary"]
        lines.append(f'| {q["layer"]} | {q["d_sae"]} | {q.get("fvu_excluding_first_token",q["fvu_centered"]):.4f} | {q["mean_l0"]:.1f} | {q["mean_kl"]:.4f} |')
    lines += ["", "## Reading the trace", "",
              "Features at position t describe the prefix through token t. The next-token probabilities on that row predict token t+1. In particular, an active feature on a supplied queen token is not evidence that it caused queen to be predicted.", "",
              "The supplied continuation is not counted as a model success. The pre-answer distribution is independently checked against a forward pass that has never seen that continuation. GPT-2's first-position activation outliers dominate pooled variance, so the main FVU table excludes each sequence's first token; all-token FVU and individual token errors remain available in the raw data.", "",
              "All active features are saved, with exact token IDs, token offsets, hook names, SAE configs, checkpoint hashes, raw residuals, and per-token relative squared reconstruction errors. SAE encoding occurs offline on the unmodified cached activations. Separate reconstruction-only forwards measure distribution distortion; they do not generate the saved rollout.", "",
              "Neuronpedia descriptions in the inspector are automatic labels, not established semantic identities. Independent gender contrasts use mother/father, sister/brother, aunt/uncle, and girl/boy at the source token. Candidate ranking excludes king/queen; their responses are reported afterwards. This small discovery set cannot establish a general gender concept or causal transport.", "",
              "Feature indices belong to one checkpoint. Equal indices at different layers or widths are unrelated. Raw coefficient magnitudes should not be compared across SAEs. Post block L is the input residual to block L+1; these checkpoints alone do not separate attention from the MLP within a block.", "",
              "No FRA path, semantic edit, or advantage over an attention mask has been demonstrated by these measurements."]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("WROTE", out / "feature_trace.html")


if __name__ == "__main__":
    main()
