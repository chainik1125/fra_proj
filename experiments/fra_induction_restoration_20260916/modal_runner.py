"""Serverless GPU runner for the paired restoration metric."""
from pathlib import Path
import gzip
import json
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App('fra-induction-restoration-20260916')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', 'transformer-lens==2.18.0', 'sae-lens==5.10.7',
                      'transformers==4.57.5', 'numpy==1.26.4', 'matplotlib')
         .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir(str(ROOT), '/work', ignore=['results', '__pycache__']))
cache = modal.Volume.from_name('fra-semantic-single-feature-cache', create_if_missing=True)
results = modal.Volume.from_name('fra-induction-restoration-results-20260916', create_if_missing=True)


@app.function(image=image, gpu='A100-80GB', memory=32768, timeout=10800, max_containers=2,
              volumes={'/cache': cache, '/results': results}, secrets=[modal.Secret.from_name('hf-token')])
def run(model_key: str, smoke: bool = False):
    import sys
    sys.path.insert(0, '/work')
    from run_restoration import run_model
    result = run_model(model_key, Path('/results'), smoke, results.commit)
    cache.commit(); results.commit()
    return result


@app.function(image=image, memory=8192, timeout=14400, volumes={'/results': results})
def coordinate(keys: list[str], smoke: bool):
    # Reattach after a coordinator restart instead of spawning duplicate GPU work.
    results.reload()
    suffix = '_smoke' if smoke else ''
    index_path = Path(f'/results/function_calls{suffix}.json')
    ids = json.loads(index_path.read_text()) if index_path.exists() else {}
    calls, completed = {}, {}
    for key in keys:
        checkpoint = Path(f'/results/{key}{suffix}.json')
        old = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if old.get('done'):
            completed[key] = old
            continue
        if key in ids:
            calls[key] = modal.FunctionCall.from_id(ids[key])
        else:
            calls[key] = run.spawn(key, smoke)
            ids[key] = calls[key].object_id
            index_path.write_text(json.dumps(ids, indent=2))
            results.commit()
    completed.update({key: call.get() for key, call in calls.items()})
    return completed


@app.local_entrypoint()
def main(model: str = 'all', smoke: bool = False):
    answers = coordinate.remote(['gpt2', 'gemma'] if model == 'all' else [model], smoke)
    (ROOT/'results').mkdir(exist_ok=True)
    for key, answer in answers.items():
        for case in answer['cases']:
            path = ROOT/'results'/f'{key}_{case["trigger"]}{"_smoke" if smoke else ""}.json.gz'
            shard = {'done': answer['done'], 'meta': answer['meta'], 'case': case}
            path.write_bytes(gzip.compress(json.dumps(shard, indent=2, allow_nan=False).encode(), compresslevel=9, mtime=0))
            print('SAVED', path, flush=True)
