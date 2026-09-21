"""Exact coefficient-mass accounting for the measured source nouns; no inference."""
import json
from pathlib import Path

from build_report import read

ROOT = Path(__file__).resolve().parent / "teacher_forced"
KEY = "gpt2-small-resid-post-v5-32k__L5"


def main():
    data = read(ROOT / "features" / f"{KEY}.json.gz")
    labels = read(ROOT / "labels.json")
    vectors = {word: dict(data["examples"][word]["features"][1]) for word in ["king", "queen"]}
    totals = {word: sum(abs(v) for v in values.values()) for word, values in vectors.items()}
    assert all(v >= 0 for x in vectors.values() for v in x.values())
    prefix = "https://www.neuronpedia.org/" + data["summary"]["neuronpedia"] + "/"
    rows = []
    for feature in vectors["king"].keys() | vectors["queen"].keys():
        url = prefix + str(feature)
        explanations = labels.get(url, {}).get("explanations", [])
        row = dict(feature=feature, url=url,
                   description=explanations[0]["description"].strip() if explanations else "No cached description",
                   description_source="Neuronpedia automatic interpretation; unvalidated")
        for word in vectors:
            value = vectors[word].get(feature, 0.0)
            row[word] = dict(activation=value, mass_share=abs(value)/totals[word])
        rows.append(row)
    rows.sort(key=lambda r: max(r[w]["mass_share"] for w in vectors), reverse=True)
    for word in vectors:
        assert abs(sum(r[word]["mass_share"] for r in rows)-1) < 1e-12
    shared = vectors["king"].keys() & vectors["queen"].keys()
    selected = {i for word in vectors for i, _ in data["examples"][word]["features"][1][:8]} | {18603, 16313, 24231}
    summary = dict(sae=data["summary"], position=1, total_mass=totals,
                   active_counts={w:len(v) for w,v in vectors.items()},
                   shared_feature_count=len(shared),
                   selected_features=sorted(selected),
                   shared_mass_share={w:sum(vectors[w][i] for i in shared)/totals[w] for w in vectors},
                   selected_table_other_share={w:1-sum(r[w]["mass_share"] for r in rows
                                                       if r["feature"] in selected) for w in vectors})
    result = dict(definition="abs(z_i) / sum_j abs(z_j); all these TopK coefficients are nonnegative",
                  summary=summary, features=rows)
    (ROOT / "activation_mass.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Feature activation mass at king and queen", "",
             "Layer 5 (zero based), post-block residual, OpenAI 32k TopK SAE loaded through SAE Lens. These are the **source nouns at token position 1** in the matched teacher-forced sentences, not the later supplied answers. Causal prefixes are exactly `The king` and `The queen`.", "",
             "Share = `abs(z_i) / sum_j abs(z_j)`. All coefficients here are nonnegative, so this is also `z_i / sum_j z_j`. Both tokens have 32 active features. Total coefficient mass is %.6f for king and %.6f for queen." % (totals["king"],totals["queen"]), "",
             "This is a partition of SAE coefficient mass. Bias, normalization mean, and reconstruction error are outside that denominator; decoder directions need not be orthogonal, so these percentages do not partition residual variance or establish causal importance. Coefficients are in the checkpoint's normalized coordinates.", "",
             "Descriptions below are Neuronpedia automatic labels, not experimentally established meanings for this sentence. Some broad labels look unrelated to the sentence. Feature 18603 has a broad people/proper-names label; the independent four-pair screen instead suggests a provisional female association. Its queen mass share is %.2f%%." % (100*next(r for r in rows if r["feature"]==18603)["queen"]["mass_share"]), "",
             f"The two tokens share {len(shared)} active feature IDs, comprising {100*summary['shared_mass_share']['king']:.2f}% of king's mass and {100*summary['shared_mass_share']['queen']:.2f}% of queen's mass. Shared activation alone does not establish which information those features carry.", "",
             "| Feature | Automatic description | king activation | king share | queen activation | queen share |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        desc = row["description"].replace("|", "\\|").replace("\n", " ")
        lines.append(f'| [{row["feature"]}]({row["url"]}) | {desc} | '
                     f'{row["king"]["activation"]:.4f} | {100*row["king"]["mass_share"]:.2f}% | '
                     f'{row["queen"]["activation"]:.4f} | {100*row["queen"]["mass_share"]:.2f}% |')
    lines += [f'| Total | | {totals["king"]:.4f} | 100% | {totals["queen"]:.4f} | 100% |', "",
              "Percentages are rounded for display; unrounded shares sum to one for each token. A zero denotes an inactive feature in this dictionary.", "",
              "[Interactive token inspector](feature_trace.html) · [Full precision data](activation_mass.json)"]
    (ROOT / "activation_mass.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row["feature"] in selected:
            print(row["feature"], row["description"], *(round(100*row[w]["mass_share"],2) for w in vectors))


if __name__ == "__main__":
    main()
