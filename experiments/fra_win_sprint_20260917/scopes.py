"""Known retrieval boundaries, without identifying which row is corrupted."""
import torch


def document_positions(tok,row):
    if 'document_positions' in row:return row['document_positions']
    text=row['text']
    if row['task']=='cards':
        start=text.index('Retrieved routing cards')
        ending='Staffing note: the Desktop team services printer hardware and handles laptop wireless onboarding.\n'
        end=text.index(ending)+len(ending)
    else:
        start=min(text.index(s) for s in ['Retrieved local support table','Retrieved local support handbook','Retrieved support table'] if s in text)
        ending='under its local contract.\n'
        end=text.index(ending)+len(ending) if ending in text else row['source_end']+1
    prefix=row['rendered_text'].index(text)
    enc=tok(row['rendered_text'],add_special_tokens=False,return_offsets_mapping=True)
    assert enc['input_ids']==row['token_ids']
    positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and b>prefix+start and a<prefix+end]
    assert set(row['source_positions']).issubset(positions)
    row['document_positions']=positions
    return positions


def mask(tok,row,device):
    m=torch.zeros((len(row['token_ids']),1),device=device);m[document_positions(tok,row)]=1
    return m
