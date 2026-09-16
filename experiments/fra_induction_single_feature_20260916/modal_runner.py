"""Run the original GPT-2 and Gemma induction tasks on two serverless GPUs."""
from pathlib import Path
import gzip
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App('fra-induction-single-feature-20260916')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', 'transformer-lens==2.18.0', 'sae-lens==5.10.7',
                      'transformers==4.57.5', 'numpy==1.26.4', 'matplotlib')
         .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir(str(ROOT), '/work', ignore=['results', '__pycache__']))
cache = modal.Volume.from_name('fra-semantic-single-feature-cache', create_if_missing=True)
results = modal.Volume.from_name('fra-induction-single-feature-results-20260916', create_if_missing=True)


@app.function(image=image, gpu='A100-80GB', timeout=10800, max_containers=2,
              volumes={'/cache': cache, '/results': results},
              secrets=[modal.Secret.from_name('hf-token')])
def run(model_key: str, smoke: bool = False):
    import sys
    sys.path.insert(0, '/work')
    from run_sweep import run_model
    answer = run_model(model_key, Path('/results'), smoke, results.commit)
    cache.commit()
    results.commit()
    return answer


@app.function(image=image, timeout=14400, volumes={'/results': results})
def coordinate(model_keys: list[str], smoke: bool):
    calls = {key: run.spawn(key, smoke) for key in model_keys}
    Path('/results/function_calls.json').write_text(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    results.commit()
    return {key: call.get() for key, call in calls.items()}


@app.local_entrypoint()
def main(model: str = 'all', smoke: bool = False):
    keys = ['gpt2', 'gemma'] if model == 'all' else [model]
    answers = coordinate.remote(keys, smoke)
    (ROOT/'results').mkdir(exist_ok=True)
    for key, answer in answers.items():
        path = ROOT/'results'/f'{key}{"_smoke" if smoke else ""}.json.gz'
        path.write_bytes(gzip.compress(json.dumps(answer, indent=2, allow_nan=False).encode(), mtime=0))
        print('SAVED', path, flush=True)
