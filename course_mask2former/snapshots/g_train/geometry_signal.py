"""Deterministic camera-frame geometry from measured depth and K only.
Normals remain in the source camera frame under image augmentation/Copy-Paste.
No world pose, object information, support mask, labels, or global RNG.
"""
import numpy as np

def geometry(depth,valid,K,region=None):
    z=np.asarray(depth,dtype=np.float32);valid=np.asarray(valid,dtype=bool)&np.isfinite(z)&(z>0)
    K=np.asarray(K,dtype=np.float64)
    assert z.ndim==2 and valid.shape==z.shape and K.shape==(3,3)
    assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2],[0,0,1]) and K[0,1]==K[1,0]==0
    z=np.where(valid,z,0);h,w=z.shape;yy,xx=np.indices((h,w),dtype=np.float32)
    xyz=np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z),-1).astype(np.float32)
    # Donors only need their exact mask bounding region. Keep global plane samples
    # identical, and retain one-pixel halo for the original normal/edge formulas.
    ids=np.flatnonzero(valid);sample=xyz.reshape(-1,3)[ids[np.linspace(0,len(ids)-1,min(2048,len(ids)),dtype=int)]] if len(ids) else np.empty((0,3))
    requested=None
    if region is not None:
        x0,y0,x1,y1=map(int,region);assert 0<=x0<x1<=w and 0<=y0<y1<=h
        sx,sy=max(0,x0-1),max(0,y0-1);ex,ey=min(w,x1+1),min(h,y1+1)
        requested=(slice(y0-sy,y1-sy),slice(x0-sx,x1-sx))
        z=z[sy:ey,sx:ex];valid=valid[sy:ey,sx:ex];xyz=xyz[sy:ey,sx:ex];h,w=z.shape
    normals=np.zeros((h,w,3),np.float32)
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
    n=np.cross(dx,dy);mag=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&(mag>1e-12)
    centre=z[1:-1,1:-1];jump=np.maximum(np.abs(z[1:-1,2:]-z[1:-1,:-2]),np.abs(z[2:,1:-1]-z[:-2,1:-1]))
    ok &= jump<=np.maximum(.01,.03*centre)
    n=np.divide(n,mag[...,None],out=np.zeros_like(n),where=(mag>1e-12)[...,None]);n=np.where((np.sum(n*xyz[1:-1,1:-1],axis=-1)>0)[...,None],-n,n)
    normals[1:-1,1:-1]=np.where(ok[...,None],n,0)
    # Deterministic, evenly sampled valid cloud; fixed 64 hypotheses and 5mm tolerance.

    best=None;best_count=0;rng=np.random.default_rng(20261013)
    if len(sample)>=96:
        for _ in range(64):
            a,b,c=sample[rng.choice(len(sample),3,replace=False)];normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
            if length<=1e-10:continue
            normal=normal/length;offset=-normal@a;inside=np.abs(sample@normal+offset)<=.005;count=int(inside.sum())
            if count>best_count:best=(normal,offset,inside);best_count=count
    height=np.zeros_like(z);plane_ok=best is not None and best_count>=max(96,int(.4*len(sample)))
    if plane_ok:
        cloud=sample[best[2]].astype(np.float64);origin=cloud.mean(0);_,_,vh=np.linalg.svd(cloud-origin,full_matrices=False);normal=vh[-1]
        if normal@origin>0:normal=-normal
        height=np.where(valid,np.clip(np.einsum('hwc,c->hw',xyz,normal)-normal@origin,-.5,.5),0).astype(np.float32)
    edge=np.zeros_like(z)
    for a,b,va,vb,dst in [(z[:,1:],z[:,:-1],valid[:,1:],valid[:,:-1],(slice(None),slice(1,None))), (z[:,:-1],z[:,1:],valid[:,:-1],valid[:,1:],(slice(None),slice(None,-1))), (z[1:],z[:-1],valid[1:],valid[:-1],(slice(1,None),slice(None))), (z[:-1],z[1:],valid[:-1],valid[1:],(slice(None,-1),slice(None)))]:
        change=np.where(va&vb,np.minimum(np.abs(a-b)/np.maximum(a,.01),1),0);edge[dst]=np.maximum(edge[dst],change)
    value=np.concatenate((np.minimum(z,10)[...,None],height[...,None],normals,edge[...,None],valid[...,None]),axis=-1).astype(np.float32)
    assert value.shape==(h,w,7) and np.isfinite(value).all()
    if requested is not None:value=np.ascontiguousarray(value[requested])
    return value,dict(valid_pixels=int(valid.sum()),normal_pixels=int(ok.sum()),plane_valid=bool(plane_ok),plane_support=best_count,plane_sample=len(sample),normal_frame='original_source_camera',depth_semantics='camera_axis_z')
