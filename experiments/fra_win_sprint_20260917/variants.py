"""Additional realistic lookup contexts, fixed before their behavioral screen."""
import data
LABELS=data.LABELS
TASKS=['contracts','contracts_short','named_offices','narrative_contracts']


def suite(task,split):
    assert task in TASKS
    base='tenants' if task=='contracts_short' else 'tenants_long'
    rows=[]
    for original in data.suite(base,split):
        r=original.copy();text=r['text'];source=text[r['source_start']:r['source_end']]
        if 'contracts' in task:
            replacements={
                'North campus | printing device | power or startup | Print':'North campus | printing device | power or startup | Desktop',
                'South campus | portable computer | wireless connectivity | Network':'South campus | portable computer | wireless connectivity | Desktop',
            }
            for a,b in replacements.items():text=text.replace(a,b)
            explanation=('Local staffing notes: the Desktop technicians service North-campus printer hardware. '
                         'The South-campus Desktop technicians also handle laptop wireless onboarding and connectivity.\n')
            text=text.replace('The South campus print-services team',explanation+'The South campus print-services team')
            if (r['tenant']==0 and r['a']==1 and r['b']==0) or (r['tenant']==1 and r['a']==0 and r['b']==1):r['expected']='Desktop'
        if task=='named_offices':
            for a,b in [('North campus','Cedar office'),('South campus','Maple office')]:
                text=text.replace(a,b);source=source.replace(a,b)
        if task=='narrative_contracts':
            def sentence(line):
                site,device,fault,queue=line.split(' | ')
                return f'At {site}, incidents involving {device}s and {fault} are handled by {queue}'
            source_new=sentence(source)
            for line in text.splitlines():
                if (line.startswith('North campus | ') or line.startswith('South campus | ')) and line.count(' | ')==3:text=text.replace(line,sentence(line))
            source=source_new
            text=text.replace('Retrieved local support table\nSite | Equipment | Problem | Queue','Retrieved local support handbook')
        start=text.index(source);end=start+len(source);target='Print' if r['poisoned'] else 'Network'
        r.update(text=text,source_start=start,source_end=end,source_label_start=end-len(target),
                 asset_end=text.index('Equipment type: '+r['device'])+len('Equipment type: '+r['device']),
                 problem_end=len(text)-len('.\nQueue:'),task=task,case_id=r['case_id'].replace(base+':',task+':'))
        rows.append(r)
    return rows
