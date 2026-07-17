import sys
sys.path.insert(0, "/tmp/claude-execution-allowed/simplex-research")
import ga

s = ga.seed_state()
print("seed pop ids:", [g["id"] for g in s["population"]])
print("deadline in hrs:", round((s["deadline"] - s["started"]) / 3600, 1))

# make_rollout on one example with the CoT genome
cot = next(g for g in s["population"] if g["id"] == "g0_cot")["genome"]
r = ga.make_rollout(ga.sports[0], cot["sys"], cot["tmpl"])
print("\nCoT rollout ok:", bool(r))
if r:
    print("CoT sample:\n", r["messages"][1]["content"][:400])

# breed test
fake_top = [{"id": "x", "genome": next(g for g in s["population"] if g["id"] == "g0_severe")["genome"],
             "fitness": 0.15, "broad": 0.08}]
kids = ga.breed(fake_top, 2)
print("\nbreed -> offspring:", len(kids))
for i, k in enumerate(kids):
    ok = all(p in k["tmpl"] for p in ("{first_half}", "{reason}", "{aligned}")) and bool(k.get("sys"))
    print(f"  child {i} valid:", ok, "| tmpl:", k["tmpl"][:90])
print("VALIDATE OK")
