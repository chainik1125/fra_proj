from pathlib import Path
import json,gzip
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('fra-win-sprint-20260917')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.6.0','transformer-lens==2.18.0','sae-lens==5.10.7','transformers==4.57.5','numpy==1.26.4','matplotlib')
       .env({'HF_HOME':'/cache/huggingface','TOKENIZERS_PARALLELISM':'false'}).add_local_dir(str(ROOT),'/work',ignore=['results','__pycache__']))
cache=modal.Volume.from_name('fra-semantic-single-feature-cache',create_if_missing=True)
results=modal.Volume.from_name('fra-win-sprint-20260917',create_if_missing=True)
@app.function(image=image,gpu='A100-80GB',memory=65536,timeout=7200,max_containers=2,volumes={'/cache':cache,'/results':results},secrets=[modal.Secret.from_name('hf-token')])
def run(stage):
    import sys;sys.path.insert(0,'/work')
    if stage in ['screen','oracle']:
        import importlib
        execute=importlib.import_module(stage).run
        result=execute(Path('/results'),results.commit)
    else:
        from search import run as execute
        assert stage.startswith('search_')
        name=stage.removeprefix('search_');smoke=name.endswith('_smoke');name=name.removesuffix('_smoke')
        result=execute(Path('/results'),results.commit,task=name,smoke=smoke)
    results.commit();cache.commit();return result
@app.local_entrypoint()
def main(stage:str='screen'):
    answer=run.remote(stage);raw=json.dumps(answer,allow_nan=False);out=ROOT/'results';out.mkdir(exist_ok=True)
    blob=gzip.compress(raw.encode(),mtime=0)
    if len(blob)<950000:(out/f'{stage}.json.gz').write_bytes(blob)
    else:
        chunks=[raw[i:i+600000] for i in range(0,len(raw),600000)]
        for i,chunk in enumerate(chunks):
            b=gzip.compress(json.dumps({'part':i,'parts':len(chunks),'json_fragment':chunk}).encode(),mtime=0)
            assert len(b)<1000000;(out/f'{stage}.part{i:03}.json.gz').write_bytes(b)
    print('SAVED',stage,len(blob),flush=True)
