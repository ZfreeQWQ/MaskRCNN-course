"""Extend native refinement to previously skipped tiny/thin masks; no label inputs."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from pycocotools import mask as mu

def component_seeds(mask,inner,support):
    """Keep erosion seeds; otherwise anchor a component at its strongest internal support."""
    seeds=inner.astype(bool).copy();visited=np.zeros_like(mask,bool)
    for y,x in zip(*np.nonzero(mask)):
        if visited[y,x]:continue
        stack=[(y,x)];visited[y,x]=True;component=[]
        while stack:
            yy,xx=stack.pop();component.append((yy,xx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if (dy or dx) and 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx]=True;stack.append((ny,nx))
        coords=np.asarray(component)
        if seeds[coords[:,0],coords[:,1]].any():continue
        scores=support[coords[:,0],coords[:,1]];best=np.isclose(scores,scores.max(),rtol=0,atol=1e-6)
        candidates=coords[best];center=coords.mean(0);pick=candidates[np.square(candidates-center).sum(1).argmin()]
        seeds[tuple(pick)]=True
    return seeds

@torch.inference_mode()
def refine_skipped(parent,original,all_wx,all_wy,device='cuda'):
    result=copy.deepcopy(parent);unchanged={p['segmentation']['counts'] for p in original}
    stats=dict(eligible=0,changed=0,guard_fallback=0)
    for p in result:
        area=int(mu.area(p['segmentation']))
        if not 0<area<=2048 or p['score']<.25 or p['segmentation']['counts'] not in unchanged:continue
        full=mu.decode(p['segmentation']);yy,xx=np.nonzero(full)
        x=max(0,int(xx.min())-4);y=max(0,int(yy.min())-4);x2=min(1024,int(xx.max())+5);y2=min(1024,int(yy.max())+5)
        old=full[y:y2,x:x2].astype(bool);m=torch.as_tensor(old.astype(np.float32),device=device)[None,None]
        inner=1-F.max_pool2d(1-m,3,1,1);outer=F.max_pool2d(m,3,1,1)
        if area>=16 and int(inner.sum())>=2:continue
        stats['eligible']+=1;wx=all_wx[:,:,y:y2,x:x2-1];wy=all_wy[:,:,y:y2-1,x:x2]
        support=m*(F.pad(wx*m[:,:,:,1:],(0,1,0,0))+F.pad(wx*m[:,:,:,:-1],(1,0,0,0))+F.pad(wy*m[:,:,1:,:],(0,0,0,1))+F.pad(wy*m[:,:,:-1,:],(0,0,1,0)))
        seeds=component_seeds(old,inner[0,0].cpu().numpy(),support[0,0].cpu().numpy())
        seeds=torch.as_tensor(seeds,device=device)[None,None]
        prior=.25+.5*m;u=prior.clone();den=.25+F.pad(wx,(0,1,0,0))+F.pad(wx,(1,0,0,0))+F.pad(wy,(0,0,0,1))+F.pad(wy,(0,0,1,0))
        for _ in range(24):
            num=.25*prior+F.pad(wx*u[:,:,:,1:],(0,1,0,0))+F.pad(wx*u[:,:,:,:-1],(1,0,0,0))+F.pad(wy*u[:,:,1:,:],(0,0,0,1))+F.pad(wy*u[:,:,:-1,:],(0,0,1,0))
            u=num/den;u=torch.where(seeds,torch.ones_like(u),torch.where(outer.bool(),u,torch.zeros_like(u)))
        new=(u[0,0]>.5).cpu().numpy();na=int(new.sum());inter=int((new&old).sum())
        if area<16:
            budget=max(1,math.ceil(.25*area));allowed=na>=1 and abs(na-area)<=budget and int((new^old).sum())<=max(2,budget)
        else:allowed=.75*area<=na<=1.25*area and inter/max(1,area+na-inter)>=.75
        if not allowed:stats['guard_fallback']+=1;continue
        if np.array_equal(new,old):continue
        full[:]=0;full[y:y2,x:x2]=new;encoded=mu.encode(np.asfortranarray(full));encoded['counts']=encoded['counts'].decode('ascii')
        p['segmentation']=encoded;stats['changed']+=1
    return result,stats
