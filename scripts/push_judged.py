# Push analyst's /tmp/judged_out/<base>/<cell>/*.json to HF <model_prefix>/<base>/<cell>/
# Model prefix derived from filename markers, with JSON peek fallback for ambiguous
# gpt4o_judged_*.json files (which carry no model marker in the name).
# Idempotent: skips files already on HF. Usage: python push_judged.py [glob-substr]
import sys, glob, os, json
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = 'dmanningcoe/fra-phase1-steering-data'
existing = set(api.list_repo_files(repo, repo_type='dataset'))
filt = sys.argv[1] if len(sys.argv) > 1 else ''
pushed = skipped = unknown = 0

def model_prefix(path: str) -> str | None:
    """Return 'qwen14b' / 'qwen7b' / None. Filename first (combined files always have
    the marker), then JSON-peek the entry's sae_id/hook_name (judged files do not name
    the model)."""
    name = os.path.basename(path)
    if 'qwen14b' in name or 'L24_' in name: return 'qwen14b'
    if 'qwen7b'  in name or 'L15_' in name: return 'qwen7b'
    try:
        d = json.load(open(path))
        if d and isinstance(d, list):
            sid = (d[0].get('sae_id') or '') + (d[0].get('hook_name') or '')
            if 'qwen14b' in sid or 'L24' in sid or 'blocks.24' in sid: return 'qwen14b'
            if 'qwen7b'  in sid or 'L15' in sid or 'blocks.15' in sid: return 'qwen7b'
    except Exception:
        pass
    return None

for f in sorted(glob.glob('/tmp/judged_out/**/*.json', recursive=True)):
    rel = f.split('/tmp/judged_out/', 1)[1]    # <base>/<cell>/<file>
    prefix = model_prefix(f)
    if prefix is None:
        print('  ?', f, '(unknown model — skipped)'); unknown += 1; continue
    tgt = f'{prefix}/{rel}'
    if filt and filt not in tgt: continue
    if tgt in existing: skipped += 1; continue
    api.upload_file(path_or_fileobj=f, path_in_repo=tgt, repo_id=repo, repo_type='dataset',
                    commit_message='judged backup (model-aware push)')
    print('  →', tgt); pushed += 1
print(f'[push] pushed {pushed}, skipped {skipped}, unknown {unknown}')
