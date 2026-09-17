"""Test a semantic AND gate before choosing any intervention or SAE feature."""
import hashlib
import json
from pathlib import Path
import time
import torch
from transformer_lens import HookedTransformer
from design import LABELS, FORMATS, suite

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def run(out_dir, commit=None):
    torch.set_grad_enabled(False); torch.set_num_threads(4); torch.manual_seed(0)
    start = time.time()
    model = HookedTransformer.from_pretrained('gemma-2-2b', device='cuda', dtype=torch.float16)
    model.eval(); tok = model.tokenizer
    label_tokens = {label: tok.encode(' '+label, add_special_tokens=False) for label in LABELS}
    assert all(len(v) == 1 for v in label_tokens.values()), label_tokens
    label_ids = [label_tokens[label][0] for label in LABELS]
    output = {'done': False, 'model': 'gemma-2-2b', 'label_token_ids': label_tokens,
              'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['design.py', 'feasibility.py']},
              'rows': []}
    for form in FORMATS:
        # Only calibration vocabulary is used to choose the presentation format.
        for row in suite(form, 'calibration'):
            ids = [tok.bos_token_id]+tok.encode(row['text'], add_special_tokens=False)
            logits = model.run_with_hooks(torch.tensor([ids], device='cuda'),
                fwd_hooks=[('ln_final.hook_normalized', lambda a, hook: a[:, -1:])])[0, -1].float()
            probs = logits.softmax(-1); restricted = logits[label_ids].softmax(-1)
            values, indices = probs.topk(5)
            output['rows'].append({**row, 'token_ids': ids,
                'probabilities': {label: float(probs[tid]) for label, tid in zip(LABELS, label_ids)},
                'restricted_probabilities': {label: float(p) for label, p in zip(LABELS, restricted)},
                'predicted_label': LABELS[int(restricted.argmax())],
                'label_mass': float(probs[label_ids].sum()),
                'top_tokens': [{'token': tok.decode([int(i)]), 'probability': float(v)} for v, i in zip(values, indices)]})
        output['runtime_s'] = time.time()-start
        path = Path(out_dir)/'feasibility.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, allow_nan=False))
        if commit: commit()
        print('FEASIBILITY', form, 'seconds', round(time.time()-start), flush=True)
    output['done'] = True
    path.write_text(json.dumps(output, indent=2, allow_nan=False))
    if commit: commit()
    return output
