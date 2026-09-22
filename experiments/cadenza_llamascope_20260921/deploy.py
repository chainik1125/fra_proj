"""Small-file-only deployment to the existing Simplex environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
CAMPAIGN = "/data/users/dmitry/sae-middle/campaigns/A-scope-overnight-20260921"
LIB_FILES = ["config.py", "train.py", "remote.py", "campaign.py", "steering.py", "restoration.py", "caa_eval.py",
             "dom_confirmation.py", "dom_layers.py", "single_eval.py", "test_pipeline.py", "requirements.txt"]


def remote(code, data=b""):
    return subprocess.run(["ssh", "simplex1", shlex.join(["python3", "-c", code])], input=data, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["sync", "start", "status", "preflight"])
    a = p.parse_args()
    if a.action == "sync":
        files = {p.name: p.read_text() for p in HERE.glob("*.py")}
        files.update({f"lib/{n}": (HERE / "baseline_lib" / n).read_text() for n in LIB_FILES})
        manifest = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()}
        plan = json.loads((HERE / "plan.json").read_text())
        assert all(len(s.encode()) < 500000 for s in files.values())
        payload = json.dumps({"files": files, "manifest": manifest, "plan": plan}).encode()
        assert len(payload) < 2000000
        code = f'''import json,sys,pathlib,hashlib
c=pathlib.Path({CAMPAIGN!r}); d=json.load(sys.stdin); c.mkdir(parents=True,exist_ok=True)
for n,s in d['files'].items():
 p=c/'src'/n; p.parent.mkdir(parents=True,exist_ok=True); assert hashlib.sha256(s.encode()).hexdigest()==d['manifest'][n]; p.write_text(s)
for n,v in [('source_manifest.json',d['manifest']),('plan.json',d['plan'])]:
 p=c/(n+'.tmp'); p.write_text(json.dumps(v,indent=2)); p.replace(c/n)
print(json.dumps({{'campaign':str(c),'source_files':len(d['files']),'payload_bytes':{len(payload)}}}))
'''
        remote(code, payload)
    elif a.action == "start":
        remote(f'''import pathlib,subprocess,json,sys
c=pathlib.Path({CAMPAIGN!r})
assert not (c/'controller_launch.json').exists(), 'Already launched'
with (c/'controller.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([sys.executable,str(c/'src/controller.py'),'--campaign',str(c)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
v={{'pid':p.pid,'campaign':str(c)}}; (c/'controller_launch.json').write_text(json.dumps(v)); print(json.dumps(v))
''')
    elif a.action == "preflight":
        remote(f'''import sys,pathlib,fcntl,subprocess
c=pathlib.Path({CAMPAIGN!r}); sys.path.insert(0,str(c/'src')); import controller
r=c.parents[1]
with open('/tmp/fra-sae-'+str(__import__('os').getuid())+'-gpu1.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); info=controller.idle_gpu(1); env=controller.environment(r,c,info['uuid'])
 subprocess.run([str(r/'venv/bin/python'),'-m','pytest','-q',str(c/'src/test_scope.py'),str(c/'src/lib/test_pipeline.py')],cwd=c/'src',env=env,check=True,timeout=300)
''')
    else:
        remote(f'''import pathlib,json,os
c=pathlib.Path({CAMPAIGN!r}); s=json.loads((c/'status.json').read_text()); pid=s['controller_pid']; s['controller_alive']=pathlib.Path('/proc/'+str(pid)+'/cmdline').exists(); print(json.dumps(s,indent=2))
''')


if __name__ == "__main__":
    main()
