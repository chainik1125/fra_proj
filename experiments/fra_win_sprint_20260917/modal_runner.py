from pathlib import Path
import json,gzip
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('fra-win-sprint-20260917')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.6.0','transformer-lens==2.18.0','sae-lens==5.10.7','transformers==4.57.5','numpy==1.26.4','matplotlib')
       .env({'HF_HOME':'/cache/huggingface','TOKENIZERS_PARALLELISM':'false'}).add_local_dir(str(ROOT),'/work',ignore=['results','__pycache__']))
cache=modal.Volume.from_name('fra-semantic-single-feature-cache',create_if_missing=True)
results=modal.Volume.from_name('fra-win-sprint-20260917',create_if_missing=True)
def execute(stage):
    import sys;sys.path.insert(0,'/work')
    if stage in ['screen','oracle','variant_screen','cards_table_screen']:
        import importlib
        execute=importlib.import_module(stage).run
        result=execute(Path('/results'),results.commit)
    elif stage.startswith('variant_search_'):
        from variant_search import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('variant_search_'))
    elif stage.startswith('variant_refine_'):
        from variant_refine import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('variant_refine_'))
    elif stage.startswith('variant_learn_pairs_'):
        from variant_learn_pairs import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('variant_learn_pairs_'))
    elif stage.startswith('multi_sae_'):
        from multi_sae import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('multi_sae_'))
    elif stage.startswith('pair_budget_'):
        from pair_budget import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('pair_budget_'))
    elif stage.startswith('robustness_'):
        from robustness import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('robustness_'))
    elif stage.startswith('mechanism_pre_distinct_'):
        from mechanism import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('mechanism_pre_distinct_'),family='fra_distinct',phase='preconfirmation')
    elif stage.startswith('mechanism_pre_'):
        from mechanism import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('mechanism_pre_'),phase='preconfirmation')
    elif stage.startswith('mechanism_distinct_'):
        from mechanism import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('mechanism_distinct_'),family='fra_distinct')
    elif stage.startswith('mechanism_'):
        from mechanism import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('mechanism_'))
    elif stage.startswith('confirm_'):
        from confirm import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('confirm_'))
    elif stage.startswith('polish_'):
        from polish import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('polish_'))
    elif stage.startswith('refine_'):
        from refine import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('refine_'))
    elif stage.startswith('learn_pairs_'):
        from learn_pairs import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('learn_pairs_'))
    elif stage.startswith('baseline_scopes_'):
        from baseline_scopes import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('baseline_scopes_'))
    elif stage.startswith('baseline_nobos_'):
        from baseline_nobos import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('baseline_nobos_'))
    elif stage.startswith('baseline_abs_'):
        from baseline_abs import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('baseline_abs_'))
    elif stage.startswith('baseline_extra2_'):
        from baseline_extra2 import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('baseline_extra2_'))
    elif stage.startswith('baseline_extra_'):
        from baseline_extra import run as execute
        result=execute(Path('/results'),results.commit,task=stage.removeprefix('baseline_extra_'))
    else:
        from search import run as execute
        assert stage.startswith('search_')
        name=stage.removeprefix('search_');smoke=name.endswith('_smoke');name=name.removesuffix('_smoke')
        result=execute(Path('/results'),results.commit,task=name,smoke=smoke)
    results.commit();cache.commit();return result
@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(65536,65536),timeout=7200,max_containers=1,volumes={'/cache':cache,'/results':results},secrets=[modal.Secret.from_name('hf-token')])
def run(stage):
    return execute(stage)

@app.function(image=image,gpu='H100!',cpu=(4,4),memory=(65536,65536),timeout=7200,max_containers=1,volumes={'/cache':cache,'/results':results},secrets=[modal.Secret.from_name('hf-token')])
def run_fast(stage):
    return execute(stage)

@app.function(image=image,cpu=.125,memory=512,timeout=60,volumes={'/results':results})
def archive_narrative_a100():
    import hashlib
    src=Path('/results/search_narrative_contracts.json');dst=Path('/results/search_narrative_contracts_a100_pilot.json')
    if not src.exists():return {'archived':False,'reason':'No checkpoint had yet been saved'}
    assert not dst.exists()
    raw=src.read_bytes();result=json.loads(raw);assert not result['done']
    src.rename(dst);results.commit()
    return {'archived':True,'path':str(dst),'sha256':hashlib.sha256(raw).hexdigest(),'points':len(result['points']),
            'reason':'Restart the short pilot on H100 with no mixed-hardware checkpoint reuse'}

@app.local_entrypoint()
def main(stage:str='screen',fast:bool=False):
    print('GPU_REQUEST', 'CPU' if stage=='archive_narrative_a100' else 'H100!' if fast else 'A100-80GB',flush=True)
    answer=archive_narrative_a100.remote() if stage=='archive_narrative_a100' else (run_fast if fast else run).remote(stage)
    raw=json.dumps(answer,allow_nan=False);out=ROOT/'results';out.mkdir(exist_ok=True)
    blob=gzip.compress(raw.encode(),mtime=0)
    if len(blob)<950000:(out/f'{stage}.json.gz').write_bytes(blob)
    else:
        chunks=[raw[i:i+600000] for i in range(0,len(raw),600000)]
        for i,chunk in enumerate(chunks):
            b=gzip.compress(json.dumps({'part':i,'parts':len(chunks),'json_fragment':chunk}).encode(),mtime=0)
            assert len(b)<1000000;(out/f'{stage}.part{i:03}.json.gz').write_bytes(b)
    print('SAVED',stage,len(blob),flush=True)
