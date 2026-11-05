"""Native 1024x1024 RGB-D mask boundary refinement, without label inputs.

rgb: float or uint8 [3,H,W], values 0..255 (RGB/BGR order immaterial).
depth_m: float32 [H,W] in meters. valid: bool [H,W].
predictions: original per-image COCO result dictionaries, including image_id.
All hyperparameters are frozen from the validation-gated experiment.
"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from pycocotools import mask as mu
from learned_edges import affinities

def encode(m):
    r=mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r['counts']=r['counts'].decode('ascii')
    return r

def merge(items):
    items=sorted(items,key=lambda p:p['score'],reverse=True)
    if not items:return []
    overlap=mu.iou([p['segmentation'] for p in items],[p['segmentation'] for p in items],[0]*len(items));keep=[]
    for i in range(len(items)):
        if not keep or max(overlap[i,keep])<.5:keep.append(i)
        if len(keep)==100:break
    return [items[i] for i in keep]

@torch.inference_mode()
def refine_and_weights(predictions,rgb,depth_m,valid,camera_k,device='cuda'):
    all_wx,all_wy=affinities(rgb,depth_m,valid,camera_k,device)
    rgb=torch.as_tensor(rgb,dtype=torch.float32,device=device)
    depth=torch.as_tensor(depth_m,dtype=torch.float32,device=device)[None]
    valid=torch.as_tensor(valid,dtype=torch.bool,device=device)[None]
    assert tuple(rgb.shape)==(3,1024,1024) and tuple(depth.shape)==(1,1024,1024)
    result=copy.deepcopy(predictions)
    for p in result:
        mask=mu.decode(p['segmentation']);area=int(mask.sum())
        if not 16<=area<=2048 or p['score']<.25:continue
        yy,xx=np.nonzero(mask);x=max(0,int(xx.min())-4);y=max(0,int(yy.min())-4);x2=min(1024,int(xx.max())+5);y2=min(1024,int(yy.max())+5)
        m=torch.as_tensor(mask[y:y2,x:x2].astype(np.float32),device=device)[None,None]
        inner=1-F.max_pool2d(1-m,3,1,1);outer=F.max_pool2d(m,3,1,1)
        if inner.sum()<2:continue
        wx=all_wx[:,:,y:y2,x:x2-1]
        wy=all_wy[:,:,y:y2-1,x:x2]
        prior=.25+.5*m;u=prior.clone();den=.25+F.pad(wx,(0,1,0,0))+F.pad(wx,(1,0,0,0))+F.pad(wy,(0,0,0,1))+F.pad(wy,(0,0,1,0))
        for _ in range(24):
            num=.25*prior+F.pad(wx*u[:,:,:,1:],(0,1,0,0))+F.pad(wx*u[:,:,:,:-1],(1,0,0,0))+F.pad(wy*u[:,:,1:,:],(0,0,0,1))+F.pad(wy*u[:,:,:-1,:],(0,0,1,0))
            u=num/den;u=torch.where(inner.bool(),torch.ones_like(u),torch.where(outer.bool(),u,torch.zeros_like(u)))
        new=(u[0,0]>.5).cpu().numpy();na=int(new.sum());inter=int((new&mask[y:y2,x:x2].astype(bool)).sum())
        if not .75*area<=na<=1.25*area or inter/(area+na-inter)<.75:continue
        full=np.zeros((1024,1024),np.uint8);full[y:y2,x:x2]=new;p['segmentation']=encode(full)
    return merge(result),all_wx,all_wy
