"""Modal runner for the compound semantic-filter experiment."""
from pathlib import Path
import gzip
import json
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App('fra-compound-semantic-20260916')
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0', 'transformer-lens==2.18.0', 'sae-lens==5.10.7',
                 'transformers==4.57.5', 'numpy==1.26.4', 'matplotlib')
    .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(str(ROOT), '/work', ignore=['results', '__pycache__']))
cache = modal.Volume.from_name('fra-semantic-single-feature-cache', create_if_missing=True)
results = modal.Volume.from_name('fra-compound-semantic-results-20260916', create_if_missing=True)


@app.function(image=image, gpu='A100-80GB', memory=32768, timeout=10800, max_containers=2,
    volumes={'/cache': cache, '/results': results}, secrets=[modal.Secret.from_name('hf-token')])
def run(stage: str):
    import sys
    sys.path.insert(0, '/work')
    if stage.startswith('feasibility'):
        from feasibility import run as execute
        answer = execute(Path('/results'), results.commit, instruction_tuned=stage != 'feasibility', larger=stage.endswith('9b'))
    else:
        from experiment import run as execute
        answer = execute(Path('/results'), results.commit, smoke=stage.endswith('smoke'))
    results.commit(); cache.commit()
    return answer


@app.local_entrypoint()
def main(stage: str = 'feasibility'):
    result = run.remote(stage)
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    blob = gzip.compress(json.dumps(result, indent=2, allow_nan=False).encode(), compresslevel=9, mtime=0)
    if len(blob) < 1_000_000:
        (out/f'{stage}.json.gz').write_bytes(blob)
    else:
        raw = json.dumps(result, allow_nan=False)
        chunks = [raw[i:i+600_000] for i in range(0,len(raw),600_000)]
        for i, chunk in enumerate(chunks):
            part = gzip.compress(json.dumps({'part':i, 'parts':len(chunks), 'json_fragment':chunk}).encode(), compresslevel=9, mtime=0)
            assert len(part) < 1_000_000
            (out/f'{stage}.part{i:03}.json.gz').write_bytes(part)
    print('SAVED', stage, len(blob), flush=True)
