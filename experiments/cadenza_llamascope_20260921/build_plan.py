"""Create the authorized queue and import existing frozen choices, without ML."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/data/users/dmitry/sae-middle"
REFERENCE = ROOT + "/runs/A-input-L08-steering-s2-20260921"
REVISIONS = {8: "8dbc1d85edfced43081c03c38b05514dbab1368b", 32: "336730e758fed3cb2273276703d836aa8659d293"}


def main():
    tasks = []
    for layer in (8, 12, 16):
        for expansion in (8, 32):
            name = f"scope-L{layer:02d}-{expansion}x"
            common = dict(attention_layer=layer, expansion=expansion, revision=REVISIONS[expansion],
                          source_kind="official", reference=REFERENCE)
            tasks.append(dict(common, id=name + "-quality", kind="diagnostic", method="single", priority=-1))
            for method, priority in (("single", 2), ("ov", 3), ("qkov", 4)):
                deps = [name + "-quality"]
                if method != "single":
                    deps.append(name + "-single")
                tasks.append(dict(common, id=name + "-" + method, kind="sweep", method=method,
                                  priority=priority, dependencies=deps))
    for method in ("single", "ov", "qkov"):
        tasks.append(dict(id="local-L12-" + method, kind="sweep", source_kind="local",
            attention_layer=12, expansion=8, method=method, priority=0, needs_training=True,
            sae_source=ROOT + "/runs/A-input-L12-100M-20260921", reference=REFERENCE))
    prior_path = HERE.parent / "cadenza_mid_sae/STEERING_RESULTS_20260921.json"
    prior = json.loads(prior_path.read_text())
    for item in prior["runs"]:
        s = item["summary"]
        if s["layer"] not in (8, 16) or "ln1.hook_normalized" not in s["hook"]:
            continue
        assert all(item["audit"].values())
        tasks.append(dict(id=f"frozen-L{s['layer']:02d}-{s['method']}", kind="frozen", source_kind="local",
            attention_layer=s["layer"], expansion=8, method=s["method"], priority=5,
            sae_source=ROOT + f"/runs/A-input4-100M-20260921-L{s['layer']:02d}-train-a1", reference=REFERENCE,
            choices={r: {"candidate": c["candidate"], "alpha": c["alpha"]} for r, c in s["results"].items()},
            frozen_source=item["remote_dir"], frozen_selected_sha256=s["selected_sha256_before_test"],
            local_audited_source_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest()))
    dom_path = HERE.parent / "cadenza_mid_sae/DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json"
    dom = json.loads(dom_path.read_text())
    for item in dom["records"]:
        if item["rule"] != "restoration" or len(item["layers"]) != 1:
            continue
        layer = item["layers"][0]
        wanted = (4, 8, 12, 16) if item["variant"] == "input_prompt" else (7, 8, 11, 12, 15, 16)
        if layer not in wanted:
            continue
        tasks.append(dict(id=f"frozen-dom-L{layer:02d}-{item['variant']}", kind="frozen", source_kind="dom",
            attention_layer=layer, method="dom_" + item["variant"], priority=5, reference=REFERENCE,
            directions=ROOT + "/campaigns/A-scope-overnight-20260921/dom_directions.json",
            directions_sha256="8e6e0f2a634492ce1173bfbffd9e8d2ecc05188fe9090ccc1268e9dd76b6f98f",
            dom_variant=item["variant"], dom_coefficient=item["coefficient"], frozen_source=item["run_id"],
            local_audited_source_sha256=hashlib.sha256(dom_path.read_bytes()).hexdigest()))
    plan = dict(started=datetime.fromisoformat("2026-09-22T04:48:12+00:00").timestamp(),
        experiment_deadline=datetime.fromisoformat("2026-09-22T15:00:00+00:00").timestamp(),
        report_deadline=datetime.fromisoformat("2026-09-22T16:00:00+00:00").timestamp(),
        host="simplex1", gpus=[0, 1, 2, 3, 4, 6, 7], aggregate_gpu_limit=8,
        training_run=ROOT + "/runs/A-input-L12-100M-20260921", tasks=tasks, sealed=False)
    assert len({t["id"] for t in tasks}) == len(tasks)
    (HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks), "experimental_deadline": "2026-09-22 08:00 PDT",
                      "report_deadline": "2026-09-22 09:00 PDT", "gpus": plan["gpus"]}))


if __name__ == "__main__":
    main()
