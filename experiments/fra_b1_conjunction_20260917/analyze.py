"""Seed-wise analysis of the unchanged collaborator sweep (standard library)."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
GRIDS = {"fra": [1, 2, 4, 8, 16, 32], "feat1": [.25, .5, 1, 2, 4, 8],
         "dom": [.25, .5, 1, 2, 4, 8], "pay": [.5, 1, 2, 4, 8],
         "oracle": [2, 4, 8, 16, 1000]}
METRICS = [f"{m}_{probe}" for m in ("colKL", "colDrop") for probe in ("reuseA", "reuseB")]


def interpolate(rows, threshold):
    origin = {"removal": 0., "coefficient": 0., **{k: 0. for k in METRICS}}
    prior = origin
    for row in rows:
        if prior["removal"] <= threshold <= row["removal"] and row["removal"] > prior["removal"]:
            weight = (threshold - prior["removal"]) / (row["removal"] - prior["removal"])
            value = {k: prior[k] + weight * (row[k] - prior[k]) for k in METRICS}
            value.update(worstKL=max(value[k] for k in METRICS[:2]),
                         worstDrop=max(value[k] for k in METRICS[2:]),
                         coefficient_interval=[prior["coefficient"], row["coefficient"]],
                         removal_interval=[prior["removal"], row["removal"]],
                         origin_bracket=prior["coefficient"] == 0.)
            return value
        prior = row
    return None


def summarize(values):
    if not values:
        return {"reached": 0, "mean_worstKL": None, "mean_worstDrop": None}
    return {"reached": len(values),
            "mean_worstKL": statistics.mean(v["worstKL"] for v in values),
            "max_worstKL": max(v["worstKL"] for v in values),
            "mean_worstDrop": statistics.mean(v["worstDrop"] for v in values),
            "max_worstDrop": max(v["worstDrop"] for v in values),
            "origin_brackets": sum(v.get("origin_bracket", False) for v in values)}


def main():
    data = json.loads((OUT / "removal" / "b1_removal.json").read_text())
    grouped = defaultdict(list)
    for row in data["rows"]:
        grouped[row["group"], row["seed"], row["method"]].append(row)
    enriched = []
    for (group, seed, method), rows in grouped.items():
        assert len(rows) == len(GRIDS[method]), (group, seed, method, len(rows))
        for coefficient, row in zip(GRIDS[method], rows):
            row["coefficient"] = coefficient
            enriched.append(row)
    cases = sorted({(g, s) for g, s, _ in grouped})
    checkpoint = json.loads((OUT / "removal" / "checkpoint.json").read_text())
    metadata = {(r["group"], r["seed"]): r for r in checkpoint["metadata"]}
    report = {"n_cases": len(cases), "raw_rows": len(enriched), "thresholds": {},
              "metadata": list(metadata.values())}
    details = []
    for threshold in (.5, .7):
        by_method = {}
        for method in GRIDS:
            values = {}
            measured = {}
            per_group = defaultdict(list)
            for group, seed in cases:
                rows = grouped[group, seed, method]
                value = interpolate(rows, threshold)
                options = [r for r in rows if r["removal"] >= threshold]
                actual = min(options, key=lambda r: max(r[k] for k in METRICS[:2])) if options else None
                if value:
                    values[group, seed] = value
                    per_group[group].append(value)
                if actual:
                    measured[group, seed] = {**actual,
                        "worstKL": max(actual[k] for k in METRICS[:2]),
                        "worstDrop": max(actual[k] for k in METRICS[2:])}
                details.append({"group": group, "seed": seed, "method": method,
                                "threshold": threshold, "interpolated": value,
                                "measured_at_least_threshold": measured.get((group, seed)),
                                "max_removal": max(r["removal"] for r in rows)})
            by_method[method] = {**summarize(list(values.values())),
                "per_group": {g: summarize(v) for g, v in per_group.items()},
                "measured_at_least_threshold": summarize(list(measured.values()))}
        both = []
        for group, seed in cases:
            f = interpolate(grouped[group, seed, "fra"], threshold)
            s = interpolate(grouped[group, seed, "feat1"], threshold)
            if f and s:
                both.append({"group": group, "seed": seed, "fra": f, "feat1": s,
                             "delta_worstKL_fra_minus_feat1": f["worstKL"] - s["worstKL"]})
        report["thresholds"][str(threshold)] = {"methods": by_method,
            "paired_fra_feat1": {"n": len(both),
                "fra_lower_worstKL": sum(r["fra"]["worstKL"] < r["feat1"]["worstKL"] for r in both),
                "fra_mean": summarize([r["fra"] for r in both]),
                "feat1_mean": summarize([r["feat1"] for r in both]), "cases": both}}
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUT / "seedwise.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    (OUT / "rows_with_coefficients.json").write_text(json.dumps(enriched, indent=2, allow_nan=False) + "\n")
    lines = ["# B1 reproduction: seed-wise comparison", "",
             f"{len(cases)} completed group/seed cases; {len(enriched)} raw sweep rows.", "",
             "Collateral is the maximum over reuseA and reuseB, measured as next-token",
             "KL(unedited || edited), in nats. First-crossing interpolation is performed",
             "within each seed, including the exact no-edit origin. Oracle collateral is",
             "excluded. See PROTOCOL.md for limitations and the original logs for the",
             "collaborator's pooled-seed summary.", ""]
    lines += ["## Removal coverage before comparing collateral", "",
              "| Group | Seed | Base target P | Max FRA removal | Max feat1 removal | Max oracle removal |",
              "|---|---:|---:|---:|---:|---:|"]
    for group, seed in cases:
        base = metadata[group, seed]["base"]["target"]
        maxima = {m: max(r["removal"] for r in grouped[group, seed, m]) for m in GRIDS}
        lines += [f"| {group} | {seed} | {base:.4f} | {maxima['fra']:.1%} | {maxima['feat1']:.1%} | {maxima['oracle']:.1%} |"]
    lines += ["", "The maximum is over the original coefficient grid. Oracle means masking",
              "the final query → target payload token edge on the nine selected induction",
              "heads; it is not an all-head or all-edge removal bound.", ""]
    for threshold, result in report["thresholds"].items():
        lines += [f"## At {float(threshold):.0%} removal", "",
                  "Each method's mean uses only its reached cases. When coverage differs,",
                  "use the common-subset comparison below to compare FRA against feat1.", "",
                  "| Method | Reached | Mean worst KL, interpolated | Mean worst payload drop | Measured ≥ threshold: mean best worst KL |",
                  "|---|---:|---:|---:|---:|"]
        for method, s in result["methods"].items():
            if method == "oracle":
                lines += [f"| oracle (removal only) | {s['reached']}/{len(cases)} | — | — | — |"]
                continue
            fmt = lambda x: "—" if x is None else f"{x:.5f}"
            lines += [f"| {method} | {s['reached']}/{len(cases)} | {fmt(s['mean_worstKL'])} | {fmt(s['mean_worstDrop'])} | {fmt(s['measured_at_least_threshold']['mean_worstKL'])} |"]
        p = result["paired_fra_feat1"]
        lines += ["", f"On the {p['n']} cases reached by both FRA and feat1, FRA has lower",
                  f"interpolated worst KL in {p['fra_lower_worstKL']} cases.", ""]
        if p["n"]:
            lines += [f"On this common subset, mean worst KL is {p['fra_mean']['mean_worstKL']:.5f}",
                      f"for FRA and {p['feat1_mean']['mean_worstKL']:.5f} for feat1.", ""]
        lines += ["| Group | Method | Reached | Mean worst KL, interpolated |",
                  "|---|---|---:|---:|"]
        for group in sorted({g for g, s in cases}):
            total = sum(g == group for g, s in cases)
            for method in ("fra", "feat1", "dom", "pay"):
                s = result["methods"][method]["per_group"].get(group, {})
                value = s.get("mean_worstKL")
                value_text = "—" if value is None else f"{value:.5f}"
                lines += [f"| {group} | {method} | {s.get('reached', 0)}/{total} | {value_text} |"]
        lines += [""]
        for method in ("fra", "feat1", "dom", "pay"):
            s = result["methods"][method]
            lines += [f"- {method}: {s.get('origin_brackets', 0)} crossings use the no-edit origin as the lower bracket."]
        lines += [""]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
