"""Check gen_tiled / tiled hooks against the original sequential sleeper.hooks path on this GPU."""
import torch, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_layers as R
from sleeper.hooks import generate_with_hooks, make_sampling_sampler, resolve_channel_deltas, build_hooks, ACTIVE_CHANNELS, compute_sae_delta, additive_steer_hook
from sleeper.model import load_sleeper_model
from sleeper.sae import load as sae_load
torch.set_grad_enabled(False)
dev="cuda"; m=load_sleeper_model(device=dev); tok=m.tokenizer
P=R.load_prompts(m,dev,64); dep_lp,dep_attn,cln_lp,cln_attn=P
def orig(hooks,seed):
    return generate_with_hooks(m,dep_lp,hooks,16,make_sampling_sampler(temperature=1.0,seed=seed,device=dev),attention_mask=dep_attn,capture_log_softmax=True)
t0,l0=orig([],0); t1,l1=R.gen_tiled(m,dep_lp,dep_attn,[],[0],dev)
print("unsteered tok equal:",torch.equal(t0.cpu(),t1.cpu()), "max lsm diff", (l0.float()-l1.cpu().float()).abs().max().item())
refs=R.refs_for(m,P,[0],dev)
for L,scheme,kind in [(0,"ov","ln1"),(2,"ov","ln1"),(1,"conventional","resid_mid"),(3,"conv_ln1","ln1")]:
    sae,_=sae_load(R.sae_path("retrain",L,kind,0),device=dev); feat=5
    alphas=[-3.0,2.5,7.0]
    hn,delta=R.feature_delta(scheme,m,sae,L,feat,dep_lp,dep_attn)
    res=R.eval_alphas(m,tok,hn,delta,alphas,P,refs,[0],dev)
    for a in alphas:
        if scheme=="ov":
            cd=resolve_channel_deltas([(feat,"V")],ACTIVE_CHANNELS["ov"],m,sae,R.ln1_hook(L),dep_lp,dep_attn,dep_attn)
            W={c:getattr(m,f"W_{c}")[L].detach() for c in "QKV"}
            hooks=build_hooks(cd,a,ACTIVE_CHANNELS["ov"],W,R.ln1_hook(L),L)
        else:
            hp=R.ln1_hook(L) if scheme=="conv_ln1" else R.resid_mid(L)
            hooks=additive_steer_hook(compute_sae_delta(m,sae,hp,feat,dep_lp,dep_attn,attention_mask=dep_attn),a,hp)
        st,sl=orig(hooks,0)
        o=R.tile_metrics(st.to(dev),sl.to(dev),refs[0],tok)
        print(L,scheme,a,"orig",{k:round(v,5) for k,v in o.items()},"tiled",{k:round(v,5) for k,v in res[a][0].items()})
