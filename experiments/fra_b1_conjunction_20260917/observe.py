"""Observe the original script at checkpoint lines, without editing its source."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

script = Path(sys.argv[1]).resolve()
out = Path(os.environ["OUTDIR"])
lines = script.read_text().splitlines()
checkpoints = {i + 1 for i, line in enumerate(lines)
               if 'print(f"  seed {seed}: base tgt' in line}
stages = {i + 1: line.strip() for i, line in enumerate(lines)
          if line.strip().startswith(("HF = fra_ph(tt)", "HFp = {", "byLp = {", 'methods["fra"]'))}
metadata = []


def atomic_json(name, data):
    tmp = out / (name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float))
    tmp.replace(out / name)


def trace(frame, event, arg):
    if frame.f_code.co_filename != str(script) or frame.f_code.co_name != "<module>":
        return None
    if event != "line":
        return trace
    g = frame.f_globals
    if frame.f_lineno in stages:
        print("[progress]", g.get("G", {}).get("name"), "seed", g.get("seed"),
              stages[frame.f_lineno], "unix", time.time(), flush=True)
    if frame.f_lineno in checkpoints:
        metadata.append({"group": g["G"]["name"], "seed": g["seed"],
                         "base": g["base"], "top_feat": g["top_feat"],
                         "heads": g["IND"], "ncells": g["ncells"],
                         "coefficients": {m: g[k] for m, k in
                             (("fra", "FC"), ("feat1", "AC"), ("dom", "DC"),
                              ("pay", "PC"), ("oracle", "MC"))}})
        atomic_json("checkpoint.json", {"rows": g["rows"], "metadata": metadata})
    return trace


sys.path.insert(0, str(script.parent.parent))
sys.settrace(trace)
try:
    runpy.run_path(str(script), run_name="__main__")
finally:
    sys.settrace(None)
