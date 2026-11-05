"""Train-supervised RGB-D adjacency affinities; inference takes no labels."""
import functools
from pathlib import Path
import numpy as np
import torch
from torch import nn

import settings
EDGE_WEIGHTS=settings.POSTPROCESS_WEIGHTS/'edge_weights.pth'
FEATURES=('rgb_abs_b','rgb_abs_g','rgb_abs_r','depth_abs_2mm','normal_abs_x','normal_abs_y','normal_abs_z','normal_valid_both','depth_valid_both','mean_depth_m')

def network():
    return nn.Sequential(nn.Linear(10,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))

def prepare(rgb,z,valid,k):
    rgb=np.asarray(rgb,dtype=np.float32).transpose(1,2,0);z=np.asarray(z,dtype=np.float32);valid=np.asarray(valid,dtype=bool)&(z>0)&np.isfinite(z)
    z=np.where(valid,z,0);yy,xx=np.indices(z.shape,dtype=np.float32)
    xyz=np.stack(((xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z),-1)
    n=np.zeros_like(xyz);nv=np.zeros_like(valid)
    a=np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1]);length=np.linalg.norm(a,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&(length>1e-12)
    for neighbor in (z[1:-1,2:],z[1:-1,:-2],z[2:,1:-1],z[:-2,1:-1]):ok &= np.abs(neighbor-z[1:-1,1:-1])<.01
    a/=np.maximum(length[...,None],1e-12);a=np.where(a[...,2:3]<0,-a,a);n[1:-1,1:-1]=np.where(ok[...,None],a,0);nv[1:-1,1:-1]=ok
    return rgb,z,n,nv,valid

def feature_pairs(prepared,axis,positions=None):
    rgb,z,n,nv,valid=prepared
    sa,sb=(np.s_[:,:-1],np.s_[:,1:]) if axis==1 else (np.s_[:-1,:],np.s_[1:,:])
    arrays=[]
    for v,scale in [(rgb,20.),(z,.002),(n,1.)]:
        aa=v[sa].reshape(-1,3 if v.ndim==3 else 1);bb=v[sb].reshape(aa.shape)
        if positions is not None:aa=aa[positions];bb=bb[positions]
        arrays.append(np.abs(aa-bb)/scale)
    for v in (nv,valid):
        aa=(v[sa]&v[sb]).reshape(-1,1)
        arrays.append(aa if positions is None else aa[positions])
    mean=((z[sa]+z[sb])*.5).reshape(-1,1);arrays.append(mean if positions is None else mean[positions])
    result=np.concatenate(arrays,axis=1).astype(np.float32)
    return np.nan_to_num(result,nan=0,posinf=10,neginf=0).clip(0,10)

@functools.lru_cache(maxsize=2)
def load_model(device):
    model=network().to(device);state=torch.load(EDGE_WEIGHTS,map_location=device,weights_only=False);assert tuple(state['features'])==FEATURES
    model.load_state_dict(state['model'],strict=True);model.eval();return model

@torch.inference_mode()
def affinities(rgb,z,valid,k,device='cuda'):
    if isinstance(rgb,torch.Tensor):rgb=rgb.detach().cpu().numpy()
    prepared=prepare(rgb,z,valid,k);model=load_model(device);result=[]
    for axis in (1,0):
        x=feature_pairs(prepared,axis);weights=[]
        for start in range(0,len(x),65536):weights.append((-model(torch.from_numpy(x[start:start+65536]).to(device)).flatten()).sigmoid())
        w=torch.cat(weights).clamp(.001,1)
        shape=(z.shape[0],z.shape[1]-1) if axis==1 else (z.shape[0]-1,z.shape[1])
        vv=(prepared[4][:,:-1]&prepared[4][:,1:]) if axis==1 else (prepared[4][:-1,:]&prepared[4][1:,:])
        w=w.reshape(shape)*torch.as_tensor(vv,device=device);result.append(w[None,None])
    return result
