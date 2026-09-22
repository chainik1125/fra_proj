"""Literal resid_post 8 follow-up: matched 32K/128K single-feature searches.

Four disjoint candidate shards per width use at most eight GPUs. Validation is
merged and choices frozen before any confirmation inference. All ML primitives
come from the frozen overnight measurement code.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/users/dmitry/sae-middle')
CAMPAIGN = ROOT / 'campaigns/A-scope-residpost8-20260922'
REFERENCE = ROOT / 'runs/A-input-L08-steering-s2-20260921'
REVISIONS = {8:'8dbc1d85edfced43081c03c38b05514dbab1368b',32:'336730e758fed3cb2273276703d836aa8659d293'}


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p,d):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2,allow_nan=False)); t.replace(p)


def worker(run):
    sys.path.insert(0,str(HERE/'lib'))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from config import Config,MODEL_REVISIONS
    from train import require_remote,reconstruction_eval,ce_eval,prepare_data
    from steering import DEFAULT_PROTOCOL
    from single_eval import rank_single_candidates,signed_grid
    from restoration import clean_reference_batches
    from caa_eval import evaluate_setting
    from shared_worker import get_splits,plumbing,evaluate_frozen
    from scope import load_scope
    require_remote(run); torch.set_num_threads(8); torch.manual_seed(42)
    task=read(run/'task.json'); expansion=task['expansion']
    cfg=Config(layer=8,hook_kind='resid_post',d_sae=4096*expansion)
    protocol={**DEFAULT_PROTOCOL,'methods':['single'],'candidates_per_method':50}
    atomic(run/'config.json',asdict(cfg))
    tok=AutoTokenizer.from_pretrained(cfg.model_name,revision=MODEL_REVISIONS['A'],token=False)
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    splits,legacy,common=get_splits(tok,cfg,protocol,REFERENCE)
    manifest={'splits':{k:[p['key'] for p in v] for k,v in splits.items()},
              'legacy_confirmation':[p['key'] for p in legacy], 'common_confirmation':[p['key'] for p in common],
              'warning':'Common third block was inspected in the overnight results; this follow-up is exploratory.'}
    atomic(run/'split_manifest.json',manifest)
    model=AutoModelForCausalLM.from_pretrained(cfg.model_name,revision=MODEL_REVISIONS['A'],token=False,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval().requires_grad_(False)
    sae=load_scope(expansion,8,REVISIONS[expansion],run)
    with torch.inference_mode():
        plumbing(model,tok,sae,cfg,splits['selection'],run)
        if task['kind']=='shard':
            candidates=rank_single_candidates(model,tok,sae,cfg,splits['selection'],protocol,run)
            candidates=candidates[task['shard']::4]
            assert len(candidates) in (12,13)
            if task['shard']==0:
                dcfg=Config(**{**asdict(cfg),'eval_per_class':32})
                pools,_,_=prepare_data(dcfg); pools={k:v[:32] for k,v in pools.items()}
                atomic(run/'quality.json',{'reconstruction':reconstruction_eval(model,tok,sae,1.,dcfg,pools),
                    'teacher_forced_ce':ce_eval(model,tok,sae,1.,dcfg,pools)})
            batches=clean_reference_batches(model,tok,sae,cfg,splits['validation'],protocol)
            baseline,base_rows=evaluate_setting(model,tok,cfg,batches,None,0,protocol,sae=sae)
            atomic(run/'validation_baseline.json',{'metrics':baseline,'rows':base_rows})
            alphas=signed_grid(protocol['alphas']);count=0
            with (run/'validation.jsonl').open('x',buffering=1) as log:
                for candidate in candidates:
                    for alpha in alphas:
                        metrics,rows=(baseline,base_rows) if alpha==0 else evaluate_setting(model,tok,cfg,batches,candidate,alpha,protocol,sae=sae)
                        record={'candidate':candidate,'alpha':alpha,'metrics':metrics}
                        log.write(json.dumps(record,allow_nan=False)+'\n');count+=1
                        atomic(run/f"validation_f{candidate['features'][0]}_a{alpha:g}.json",{**record,'rows':rows})
                        atomic(run/'progress.json',{'event':'validation','completed':count,'target':len(candidates)*len(alphas)})
            atomic(run/'summary.json',{'state':'complete','kind':'validation_shard','records':count,'no_test_evaluation':True})
        else:
            assert task['kind']=='final'
            choices=task['choices'];atomic(run/'selected.json',choices);frozen=sha(run/'selected.json')
            results={}
            for name,pairs in [('test',splits['test']),('legacy_confirmation',legacy),('confirmation',common)]:
                results[name]=evaluate_frozen(model,tok,sae,cfg,pairs,protocol,run,choices,name)
                assert sha(run/'selected.json')==frozen
            atomic(run/'summary.json',{'state':'complete','residual_layer':8,'width':cfg.d_sae,
                'method':'single','selected_sha256_before_test':frozen,**results,
                'warning':manifest['warning']})


def merge_shards(campaign,expansion,jobs):
    sys.path.insert(0,str(HERE/'lib'))
    from single_eval import choose_single_settings
    grid=[];selection=None;split=None;baseline=None
    for shard in range(4):
        j=next(j for j in jobs if j['id']==f'{expansion}x-shard{shard}')
        assert j['state']=='complete'
        run=Path(j['run']);s=read(run/'selection.json');m=read(run/'split_manifest.json')
        b=read(run/'validation_baseline.json')
        if selection is None:selection,split,baseline=s,m,b
        assert selection==s and split==m and baseline==b, 'Shard ranking/splits/baseline diverged'
        records=[json.loads(l) for l in (run/'validation.jsonl').read_text().splitlines()]
        expected={tuple(c['features']) for c in s['candidates'][shard::4]}
        assert {tuple(r['candidate']['features']) for r in records}==expected
        grid.extend(records)
    from single_eval import signed_grid
    from steering import DEFAULT_PROTOCOL
    identities={(tuple(r['candidate']['features']),r['alpha']) for r in grid}
    expected={(tuple(c['features']),a) for c in selection['candidates'] for a in signed_grid(DEFAULT_PROTOCOL['alphas'])}
    assert len(grid)==len(identities)==750 and identities==expected
    order={tuple(c['features']):i for i,c in enumerate(selection['candidates'])}
    grid.sort(key=lambda r:(order[tuple(r['candidate']['features'])],r['alpha']))
    choices=choose_single_settings(grid)
    atomic(campaign/f'{expansion}x-selection_audit.json',{'records':750,'candidates':50,'four_disjoint_shards':True,
        'identical_rankings_splits_baselines':True,'choices':choices,
        'validation_hashes':{j['id']:sha(Path(j['run'])/'validation.jsonl') for j in jobs if j['id'].startswith(f'{expansion}x-shard')}})
    return choices


def coordinator(campaign):
    sys.path.insert(0,str(HERE/'lib'))
    from remote import idle_gpu
    plan=read(campaign/'plan.json');deadline=plan['deadline'];active={}
    tasks={f'{e}x-shard{s}':{'kind':'shard','expansion':e,'shard':s} for e in (8,32) for s in range(4)}
    state={'state':'running','pid':os.getpid(),'started':time.time(),'jobs':[],'gpu_cap':8}
    try:
        while time.time()<deadline:
            known={j['id'] for j in state['jobs']}
            for name in tasks:
                if name not in known:state['jobs'].append({'id':name,'state':'pending','attempts':0})
            for gpu,(j,proc) in list(active.items()):
                if proc.poll() is None:continue
                result=read(Path(j['run'])/'status.json')
                if result['state']=='complete':j['state']='complete'
                else:j.update(state='pending' if j['attempts']<2 else 'failed',error=result.get('error'))
                j['ended']=time.time();del active[gpu]
            for e in (8,32):
                name=f'{e}x-final'
                if name not in tasks and all(any(j['id']==f'{e}x-shard{s}' and j['state']=='complete' for j in state['jobs']) for s in range(4)):
                    tasks[name]={'kind':'final','expansion':e,'choices':merge_shards(campaign,e,state['jobs'])}
            for j in state['jobs']:
                if j['state']!='pending':continue
                for gpu in range(8):
                    if gpu in active:continue
                    try:idle_gpu(gpu)
                    except (RuntimeError,subprocess.SubprocessError):continue
                    j['attempts']+=1;run=campaign/f"{j['id']}-a{j['attempts']}";run.mkdir()
                    shutil.copytree(campaign/'src',run/'src');shutil.copy2(campaign/'source_manifest.json',run/'source_manifest.json')
                    atomic(run/'task.json',tasks[j['id']])
                    with (run/'worker.log').open('ab',buffering=0) as log:
                        proc=subprocess.Popen([sys.executable,str(run/'src/controller.py'),'--job',str(run),'--gpu',str(gpu),'--deadline',str(deadline)],
                            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                    j.update(state='running',gpu=gpu,run=str(run),worker_pid=proc.pid,started=time.time());active[gpu]=(j,proc);break
            state.update(updated=time.time(),active_gpus=len(active));assert len(active)<=8
            atomic(campaign/'status.json',state)
            finals=[j for j in state['jobs'] if j['id'].endswith('-final')]
            if len(finals)==2 and all(j['state']=='complete' for j in finals):
                state['state']='complete';break
            if any(j['state']=='failed' for j in state['jobs']):raise RuntimeError('Task failed after two attempts')
            time.sleep(10)
        else:state['state']='deadline_reached'
    except BaseException as exc:
        import traceback;traceback.print_exc();state.update(state='failed',error=str(exc))
    finally:
        for j,proc in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                j['state']='stopped'
        state.update(ended=time.time(),active_gpus=0);atomic(campaign/'status.json',state)
        if state['state']=='complete':
            result=collect(campaign);atomic(campaign/'results.json',result)
            (campaign/'summary.md').write_text(report(result))


def collect(campaign):
    sys.path.insert(0,str(HERE/'lib'))
    from single_eval import choose_single_settings
    state=read(campaign/'status.json');result={'status':state,'cells':[]}
    for e in (8,32):
        jobs=[j for j in state['jobs'] if j['id']==f'{e}x-final' and j['state']=='complete']
        if not jobs:continue
        run=Path(jobs[0]['run']);s=read(run/'summary.json');selected=read(run/'selected.json')
        audit=read(campaign/f'{e}x-selection_audit.json')
        assert sha(run/'selected.json')==s['selected_sha256_before_test'] and selected==audit['choices']
        all_records=[];rank=None
        for shard in range(4):
            j=next(j for j in state['jobs'] if j['id']==f'{e}x-shard{shard}')
            directory=Path(j['run']);rank=read(directory/'selection.json')['candidates']
            assert sha(directory/'validation.jsonl')==audit['validation_hashes'][j['id']]
            all_records.extend(json.loads(line) for line in (directory/'validation.jsonl').read_text().splitlines())
            manifest=read(directory/'source_manifest.json')
            assert all(sha(directory/'src'/k)==v for k,v in manifest.items())
        order={tuple(c['features']):i for i,c in enumerate(rank)}
        all_records.sort(key=lambda r:(order[tuple(r['candidate']['features'])],r['alpha']))
        assert choose_single_settings(all_records)==selected,'Selection differs from canonical unsharded order'
        audit={**audit,'canonical_unsharded_choices_reproduced':True,'all_shard_source_hashes_verified':True}
        hashes=read(run/'source_manifest.json');assert all(sha(run/'src'/k)==v for k,v in hashes.items())
        cell={'width':e*4096,'run':str(run),'summary':s,'audit':audit,'sae_source':read(run/'sae_source.json'),'rows':{}}
        q=next(j for j in state['jobs'] if j['id']==f'{e}x-shard0')
        cell['quality']=read(Path(q['run'])/'quality.json')
        for rule in selected:
            rows=read(run/f'confirmation_{rule}.json')['rows'];assert [r['key'] for r in rows]==s['confirmation']['keys']
            cell['rows'][rule]=[{k:v for k,v in r.items() if k=='key' or isinstance(v,(float,int,bool))} for r in rows]
        result['cells'].append(cell)
    return result


def report(result):
    lines=['# Literal resid_post 8: Llama Scope single-feature steering','',
        'Same frozen measurement code, top 50 diff-ranked features, 15 signed strengths, and 24 validation pairs. Both selection rules freeze before test.',
        'The 64-pair common confirmation set was already inspected in the overnight study; this is an exploratory follow-up.',
        '', '| Width | Rule | Feature | α | JSD (bits) | Clean preserved /64 | Clean drift | IHY removed /64 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for cell in result['cells']:
        for rule,d in cell['summary']['confirmation']['results'].items():
            m=d['metrics'];lines.append(f"| {cell['width']//1024}K | {rule} | {d['candidate']['features'][0]} | {d['alpha']:g} | {m['triggered_to_clean_js_bits']:.6f} | {round(64*m['clean_exact_match'])} | {m['clean_drift_js_bits']:.6g} | {m['escaped_phrase_count']} |")
    lines+=['','All edits apply directly to `blocks.8.hook_resid_post`, with the inherited prompt-only BOS/EOS/PAD mask. No FRA evaluation is part of this follow-up.',
            'Each width has 750 validation records, verified as four disjoint shards with identical full rankings, split identities and unsteered baselines.',
            f"Remote artifacts: `{CAMPAIGN}`. Model/checkpoint hashes, original-test and legacy-confirmation results are in `results.json`."]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','collect','coordinate','worker']);p.add_argument('--run');a=p.parse_args()
    if a.action=='worker':return worker(Path(a.run))
    if a.action=='coordinate':return coordinator(CAMPAIGN)
    if a.action=='launch':
        files={'rp8.py':Path(__file__).read_text(),'scope.py':(HERE/'scope.py').read_text(),
               'shared_worker.py':(HERE/'worker.py').read_text(),'controller.py':(HERE/'controller.py').read_text(),
               'worker.py':'import sys\nfrom rp8 import worker\nfrom pathlib import Path\nworker(Path(sys.argv[2]))\n'}
        files.update({f'lib/{f.name}':f.read_text() for f in (HERE/'baseline_lib').iterdir() if f.is_file()})
        payload={'files':files,'hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()},
                 'plan':{'residual_layer':8,'widths':[32768,131072],'max_gpus':8,'deadline':time.time()+7200}}
        code=f'''import json,sys,pathlib,subprocess,hashlib
c=pathlib.Path({str(CAMPAIGN)!r});assert not c.exists(),'Campaign already exists';d=json.load(sys.stdin);c.mkdir()
for n,s in d['files'].items():
 p=c/'src'/n;p.parent.mkdir(exist_ok=True,parents=True);assert hashlib.sha256(s.encode()).hexdigest()==d['hashes'][n];p.write_text(s)
for n,v in [('source_manifest.json',d['hashes']),('plan.json',d['plan'])]:(c/n).write_text(json.dumps(v,indent=2))
with (c/'controller.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([sys.executable,str(c/'src/rp8.py'),'coordinate'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({{'pid':p.pid,'campaign':str(c)}}))
'''
        subprocess.run(['ssh','simplex1',shlex.join(['python3','-c',code])],input=json.dumps(payload).encode(),check=True)
    elif a.action=='status':
        code=f'''import json,pathlib,time
c=pathlib.Path({str(CAMPAIGN)!r});s=json.loads((c/'status.json').read_text());print(s['state'],'age',round(time.time()-s['updated'],1))
for j in s['jobs']:
 p=pathlib.Path(j['run'])/'progress.json' if 'run' in j else None;d=json.loads(p.read_text()) if p and p.exists() else {{}};print(j['id'],j['state'],d.get('completed'),d.get('target'),j.get('error',''))
'''
        subprocess.run(['ssh','simplex1',shlex.join(['python3','-c',code])],check=True)
    else:
        code=f"import pathlib,sys,json;sys.path.insert(0,{str(CAMPAIGN/'src')!r});from rp8 import collect;print(json.dumps(collect(pathlib.Path({str(CAMPAIGN)!r}))))"
        raw=subprocess.check_output(['ssh','simplex1',shlex.join(['python3','-c',code])]);assert len(raw)<1_000_000
        result=json.loads(raw);out=HERE/'resid_post_8';out.mkdir(exist_ok=True)
        atomic(out/'results.json',result);(out/'summary.md').write_text(report(result))
        print(json.dumps({'state':result['status']['state'],'complete_cells':len(result['cells']),'output':str(out)}))


if __name__=='__main__':main()
