"""Recompute hookpoint SAE FVU directly from cached activations, once per token."""
import json
from pathlib import Path
import torch
from sae_models import TopKSAE
R=Path('/archive/fra_table3_retrain_20260925')
fresh=torch.load(R/'fresh_holdout.pt',map_location='cpu',weights_only=True)
data=json.loads((R/'sae_loss_recovered.json').read_text())
clean=~fresh['is_deployment']
torch.set_num_threads(8)
for hid in range(12):
    hook=data['hooks'][str(hid)]
    x=fresh['acts'][clean,:,hid,:].reshape(-1,768).float().cuda()
    mu=x.mean(0)
    den=float((x-mu).square().sum())
    for row in hook['seeds']:
        seed=row['seed']
        p=R/'weights'/f"sae_L{hid//3}_{hook['kind']}_s{seed}.pt"
        sae=TopKSAE(768,1536,32).cuda().eval()
        sae.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['state_dict'])
        num=0.
        with torch.inference_mode():
            for start in range(0,len(x),4096):
                y,_=sae(x[start:start+4096])
                num+=float((y-x[start:start+4096]).square().sum())
        row['activation_one_minus_fvu']=1-num/den
    hook['activation_variance_sum']=den
    print(hid,hook['kind'],[round(s['activation_one_minus_fvu'],4) for s in hook['seeds']],flush=True)
data['activation_fvu_definition']='Direct SAE reconstruction of one cached activation per token; channelwise mean baseline on clean fresh holdout'
(R/'sae_loss_recovered.json').write_text(json.dumps(data,indent=2)+'\n')
