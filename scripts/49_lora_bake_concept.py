"""TRACK A -- bake a concept->payload association into the WEIGHTS with LoRA, then it can be
attacked by the same semantic-filter pipeline (script 47) to test whether FRA still wins.

RESEARCH CONTEXT: defensive interpretability research (see docs/insen/research_context.md). Benign
placeholder association (e.g. ship->anchor). The point is to move the association from the PROMPT
(where rungs 1-5 planted it) into the model's WEIGHTS, so the removal test is on a trained-in
behaviour, not a planted one -- the main caveat of the semantic-filter result.

Design:
- Train on the BASE trigger word only ("Remember the password: ship" -> " anchor"), with varied
  filler so the model learns the association, not one context. Synonyms are HELD OUT of training.
- After training, two questions:
    (1) does the weight-baked association GENERALISE to held-out synonyms (vessel/boat/yacht)?
        -- i.e. is there a concept, not just a memorised token, now in the weights.
    (2) [script 47, separately] can FRA remove it with less collateral than the baselines?
- Merge LoRA and save to $OUTDIR/merged so transformer_lens can load it for the pipeline.

Env: needs peft + transformers (NOT the pinned FRA env; use the training env). The FRA pipeline
step runs later in fra_pin, loading the merged model.

Run under SLURM with a >=40GB GPU. Env: CONCEPT (default vessel), STEPS, LORA_R, OUTDIR.
"""
import os, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

OUT = os.environ.get("OUTDIR", "."); os.makedirs(OUT, exist_ok=True)
CONCEPT = os.environ.get("CONCEPT", "vessel")
STEPS = int(os.environ.get("STEPS", "400"))
LORA_R = int(os.environ.get("LORA_R", "16"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

# concept: (base trigger trained on, payload, held-out synonyms tested but NEVER trained)
CONCEPTS = {
    "vessel":  (" ship",   " anchor",   [" ships", " boat", " vessel", " yacht"]),
    "medical": (" doctor",  " hospital", [" nurse", " surgeon", " medicine", " patient"]),
}
plant, payload, syns = CONCEPTS[CONCEPT]

FILLER = ["The weather was mild.", "People arrived late.", "The library opened early.",
 "A dog waited by the door.", "The report took long.", "The train was quiet.",
 "She left a note.", "The garden needed water.", "Students argued.", "The coffee was cold.",
 "Chairs were stacked.", "He forgot his umbrella.", "The schedule starts Monday.",
 "Rain began after lunch.", "The shop closed.", "They walked by the lake.",
 "The printer ran out.", "The soup was salty.", "The museum added a room.", "Her sister called."]

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16).to(dev)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

Tid = tok.encode(plant, add_special_tokens=False)
Pid = tok.encode(payload, add_special_tokens=False)
assert len(Tid) == 1 and len(Pid) == 1, "trigger/payload must be single tokens"
Pid = Pid[0]


def train_example(rng):
    fs = rng.sample(FILLER, rng.randint(2, 5))
    text = " ".join(fs) + " Remember the password:" + plant
    ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
    return ids


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
rng = random.Random(0)
model.train()
for step in range(STEPS):
    ids = train_example(rng)
    x = torch.tensor(ids, device=dev).unsqueeze(0)
    tgt = torch.tensor([[Pid]], device=dev)
    full = torch.cat([x, tgt], 1)
    out = model(full)
    # loss only on the payload token
    logp = torch.log_softmax(out.logits[0, -2], -1)
    loss = -logp[Pid]
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"step {step:4d} loss {loss.item():.4f}", flush=True)

# ---- evaluate weight-baked generalisation ----
model.eval()
res = {}
with torch.no_grad():
    for w in [plant] + syns:
        ps = []
        for s in range(8):
            fs = random.Random(1000 + s).sample(FILLER, 4)
            ids = [tok.bos_token_id] + tok.encode(" ".join(fs) + " Remember the password:" + w, add_special_tokens=False)
            lg = model(torch.tensor(ids, device=dev).unsqueeze(0)).logits[0, -1].float()
            ps.append(torch.softmax(lg, -1)[Pid].item())
        res[w.strip()] = sum(ps) / len(ps)
        print(f"  P({payload.strip()} | ...:{w.strip()}) = {res[w.strip()]:.3f}", flush=True)

merged = model.merge_and_unload()
merged.save_pretrained(os.path.join(OUT, "merged")); tok.save_pretrained(os.path.join(OUT, "merged"))
json.dump({"concept": CONCEPT, "plant": plant.strip(), "payload": payload.strip(),
           "weightbaked_generalisation": res, "steps": STEPS, "lora_r": LORA_R},
          open(os.path.join(OUT, "bake_result.json"), "w"), indent=2)
print(f"\nsaved merged model to {OUT}/merged", flush=True)
print("DONE bake", flush=True)
