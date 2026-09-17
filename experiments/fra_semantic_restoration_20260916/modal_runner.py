"""Checkpointed Modal GPU harness for semantic-filter restoration."""
from pathlib import Path
import gzip
import json
import modal
ROOT = Path(__file__).resolve().parent
CONCEPTS = ['vessel', 'vehicle', 'bird', 'fire', 'war', 'medical']
app = modal.App('fra-semantic-restoration-20260916')
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0', 'transformer-lens==2.18.0', 'sae-lens==5.10.7',
                 'transformers==4.57.5', 'numpy==1.26.4', 'matplotlib')
    .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(str(ROOT), '/work', ignore=['results', '__pycache__']))
cache = modal.Volume.from_name('fra-semantic-single-feature-cache', create_if_missing=True)
results = modal.Volume.from_name('fra-semantic-restoration-results-20260916', create_if_missing=True)


@app.function(image=image, gpu='A100-80GB', memory=32768, timeout=10800, max_containers=2,
    volumes={'/cache': cache, '/results': results}, secrets=[modal.Secret.from_name('hf-token')])
def run(concept: str, smoke: bool = False):
    import sys
    sys.path.insert(0, '/work')
    from run_restoration import run_concept
    result = run_concept(concept, Path('/results'), smoke, results.commit)
    cache.commit(); results.commit()
    return result


@app.function(image=image, memory=8192, timeout=18000, volumes={'/results': results})
def coordinate(concepts: list[str], smoke: bool):
    results.reload()
    suffix = '_smoke' if smoke else ''
    index = Path(f'/results/function_calls{suffix}.json')
    ids = json.loads(index.read_text()) if index.exists() else {}
    calls, completed = {}, {}
    for c in concepts:
        path = Path(f'/results/{c}{suffix}.json')
        old = json.loads(path.read_text()) if path.exists() else {}
        if old.get('done'):
            completed[c] = old
        elif c in ids:
            calls[c] = modal.FunctionCall.from_id(ids[c])
        else:
            calls[c] = run.spawn(c, smoke); ids[c] = calls[c].object_id
            index.write_text(json.dumps(ids, indent=2)); results.commit()
    for c, call in calls.items():
        completed[c] = call.get()
        print('COLLECTED', c, flush=True)
    return completed


def save(result):
    # Each part retains all input metadata; configurations are split to respect
    # the repository's size hook without discarding per-token measurements.
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    concept = result['meta']['concept']
    name = concept+('_smoke' if result['meta']['smoke'] else '')
    points = result['points']
    for n, start in enumerate(range(0, len(points), 100)):
        shard = {**result, 'points': points[start:start+100], 'point_offset': start, 'total_points': len(points)}
        blob = gzip.compress(json.dumps(shard, indent=2, allow_nan=False).encode(), compresslevel=9, mtime=0)
        assert len(blob) < 1_000_000
        path = out/f'{name}.part{n:02d}.json.gz'; path.write_bytes(blob)
        print('SAVED', path, len(blob), flush=True)


@app.local_entrypoint()
def main(concept: str = 'all', smoke: bool = False):
    answers = coordinate.remote(CONCEPTS if concept == 'all' else [concept], smoke)
    for answer in answers.values(): save(answer)
