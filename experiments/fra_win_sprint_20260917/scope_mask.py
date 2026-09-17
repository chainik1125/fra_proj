"""Scope choices for confirmation and coefficient refinement."""
import torch
from scopes import mask as document_mask


def scope_mask(tok,row,device,scope):
    if scope=='all':return 1.
    if scope=='document':return document_mask(tok,row,device)[None]
    assert scope=='no_bos'
    m=torch.ones((1,len(row['token_ids']),1),device=device);m[:,0]=0
    return m
