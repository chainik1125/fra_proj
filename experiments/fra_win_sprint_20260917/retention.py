"""Separate clean-model mistakes from errors introduced by an intervention."""

def agreement(edited_rows,clean_rows):
    clean={r['case_id']:r for r in clean_rows};assert set(clean)=={r['case_id'] for r in edited_rows}
    out={}
    for group in ['all','joint','controls','shared_conjunction']:
        rows=[r for r in edited_rows if group=='all' or (group=='joint' and r['joint']) or
              (group=='controls' and not r['joint']) or (group=='shared_conjunction' and r['corner']=='111')]
        if not rows:continue
        cc=[clean[r['case_id']] for r in rows]
        n=sum(c['correct'] for c in cc);ntop=sum(c['top_correct'] for c in cc)
        preserved=sum(c['correct'] and r['correct'] for c,r in zip(cc,rows))
        preserved_top=sum(c['top_correct'] and r['top_correct'] for c,r in zip(cc,rows))
        out[group]={'n':len(rows),'clean_correct':n,'edited_correct':sum(r['correct'] for r in rows),
            'introduced_errors':n-preserved,'fixed_clean_errors':sum(not c['correct'] and r['correct'] for c,r in zip(cc,rows)),
            'retention_of_clean_correct':preserved/n if n else None,
            'retention_of_clean_top_correct':preserved_top/ntop if ntop else None,
            'queue_agreement':sum(c['predicted']==r['predicted'] for c,r in zip(cc,rows))/len(rows),
            'full_token_agreement':sum(c['top_token']==r['top_token'] for c,r in zip(cc,rows))/len(rows)}
    return out
