"""Bounded paired geometry adaptation of our highres42000 MaskDINO checkpoint."""
import argparse,json,math,os,random,time,traceback
from pathlib import Path
import train_maskdino_extra as t
from geometry_fusion import install
torch,np,d=t.torch,t.np,t.d
CHECKPOINT=t.ENV/'rgb_init.pth'
EXPECTED='4ab4d45d64b6400f0c52c16a3d2fe4c8e1b3c10f6b69dbc3a8f19dc7635fc2cd'

def seed():torch.manual_seed(121);torch.cuda.manual_seed_all(121);random.seed(121);np.random.seed(121)
def rng_state():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])

def build(out,arm):
    assert d.sha(CHECKPOINT)==EXPECTED
    seed();model,cfg=t.build(out,'rgb')
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items());del ck
    if arm=='geometry':install(model)
    return model,cfg

def make_optimizer(model,cfg):
    opt=d.optimizer(model,cfg);names={id(p):n for n,p in model.named_parameters()};rates=[];summary={}
    for group in opt.param_groups:
        name=names[id(group['params'][0])]
        category='new_geometry' if name.startswith('backbone.') and not name.startswith('backbone.rgb.') and hasattr(model.backbone,'rgb') else 'pretrained_backbone' if name.startswith('backbone.') else 'decoder'
        lr={'new_geometry':1e-4,'pretrained_backbone':1e-6,'decoder':1e-5}[category]
        group['lr']=lr;rates.append(lr)
        record=summary.setdefault(category,dict(lr=lr,parameters=0));record['parameters']+=group['params'][0].numel()
    return opt,rates,summary

def save(path,model,opt,step,state):
    tmp=path.with_suffix('.partial');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,run_state=state,rng=rng_state()),tmp);tmp.replace(path)
