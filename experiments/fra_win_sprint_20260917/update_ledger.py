"""Persist app history even after old stopped apps leave the CLI's listing."""
import datetime,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
dest=ROOT/'COMPUTE_LEDGER.json'
old=json.loads(dest.read_text()) if dest.exists() else {'apps':[]}
history={a['app_id']:a for a in old['apps']};stages={}
history.setdefault('ap-fSj6scMc3cqCgH594SXONI',{'app_id':'ap-fSj6scMc3cqCgH594SXONI','stage':'screen','state':'stopped','tasks':'0',
    'created_at':'2026-09-17T17:22:28+00:00','stopped_at':'2026-09-17T17:28:07+00:00','app_wall_hours':339/3600})
for path in (ROOT/'results').glob('*.log'):
    match=re.search(r'ap-[A-Za-z0-9]+',path.read_text())
    if match:stages[match.group()]=path.stem
apps=json.loads(subprocess.check_output(['modal','app','list','--json']))
now=datetime.datetime.now(datetime.timezone.utc)
for app in apps:
    if app['Description']!='fra-win-sprint-20260917':continue
    start=datetime.datetime.fromisoformat(app['Created at']);stop=datetime.datetime.fromisoformat(app['Stopped at']) if app['Stopped at'] else now
    history[app['App ID']]={'app_id':app['App ID'],'stage':stages.get(app['App ID']),'state':app['State'],'tasks':app['Tasks'],
        'created_at':start.isoformat(),'stopped_at':app['Stopped at'],'app_wall_hours':(stop-start).total_seconds()/3600}
ledger={'updated_utc':now.isoformat(),'app_wall_time_includes_queue_and_is_not_billed_gpu_time':True,
    'planning_rate_per_running_gpu_hour':3.198528,'apps':list(history.values()),
    'total_app_wall_hours':sum(a['app_wall_hours'] for a in history.values())}
dest.write_text(json.dumps(ledger,indent=2))
print('App-wall hours (including queue):',round(ledger['total_app_wall_hours'],3))
print('Active:',[(a['stage'],a['app_id'],a['tasks']) for a in history.values() if a['state']!='stopped'])
