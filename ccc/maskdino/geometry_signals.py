"""Depth-only camera geometry. No instance labels or scene/object metadata inputs.

Seven-channel geometric representation is a shared RGB-D design, also used by
the referenced teammate branches. This implementation was written independently.
Plane-relative height is an estimated geometric cue, not ground-truth table height.
"""
import gzip,json
from pathlib import Path
import numpy as np

CHANNELS=('z','plane_height','normal_x','normal_y','normal_z','relative_edge','valid')

def camera_records(root,split):
    import settings
    records={}
    if split=='test':
        raw=json.loads(settings.IMAGE_METADATA.read_text())
        assert set(raw).issubset({'images','categories','info','licenses'})
        groups=[raw['images']]
    else:
        assert split in ('train','val')
        groups=[]
        for path in sorted((Path(root)/'annotations'/split).glob('*.json.gz')):
            with gzip.open(path,'rt') as stream:groups.append(json.load(stream)['images'])
    for rows in groups:
        for row in rows:
            assert row['depth_semantics']=='camera_axis_z' and row['id'] not in records
            k=np.asarray(row['camera']['K'],dtype=np.float32)
            assert k.shape==(3,3) and k[0,0]>0 and k[1,1]>0
            records[row['id']]=k
    assert records
    return records

def xyz_from_depth(z,k):
    yy,xx=np.indices(z.shape,dtype=np.float32)
    return np.stack(((xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z),axis=-1)

def support_plane(xyz,valid,seed):
    points=xyz[::16,::16][valid[::16,::16]].astype(np.float64)
    if len(points)<64:return None,0.
    rng=np.random.default_rng(int(seed)%2**32)
    triplets=np.stack([rng.choice(len(points),3,replace=False) for _ in range(96)])
    tri=points[triplets];normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    length=np.linalg.norm(normals,axis=1);good=length>1e-8
    if not good.any():return None,0.
    normals=normals[good]/length[good,None];bias=-(normals*tri[good,0]).sum(1)
    errors=np.abs(points@normals.T+bias[None]);counts=(errors<.003).sum(0)
    best=int(np.argmax(counts));support=float(counts[best]/len(points))
    if support<.25:return None,support
    inliers=points[errors[:,best]<.003];center=inliers.mean(0)
    _,_,vt=np.linalg.svd(inliers-center,full_matrices=False);n=vt[-1]
    # Normal points towards the camera: nearer objects have positive height.
    if np.dot(n,center)>0:n=-n
    return (n.astype(np.float32),float(-np.dot(n,center))),support

def geometry(z,valid,k,seed):
    z=np.asarray(z,dtype=np.float32);valid=np.asarray(valid,dtype=bool)&np.isfinite(z)&(z>0)
    z=np.where(valid,z,0);xyz=xyz_from_depth(z,k)
    normal=np.zeros_like(xyz);normal_valid=np.zeros_like(valid)
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
    n=np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&(length>1e-12)
    center=z[1:-1,1:-1]
    for neighbor in (z[1:-1,2:],z[1:-1,:-2],z[2:,1:-1],z[:-2,1:-1]):ok &= np.abs(neighbor-center)<.01
    n=n/np.maximum(length[...,None],1e-12);n=np.where(n[...,2:3]<0,-n,n)
    normal[1:-1,1:-1]=np.where(ok[...,None],n,0);normal_valid[1:-1,1:-1]=ok
    edge=np.zeros_like(z)
    for axis in (0,1):
        a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1);a,b=tuple(a),tuple(b)
        jump=np.where(valid[a]&valid[b],np.abs(z[a]-z[b])/np.maximum((z[a]+z[b])*.5,1e-6),0)
        edge[a]=np.maximum(edge[a],jump);edge[b]=np.maximum(edge[b],jump)
    edge=np.minimum(edge,1.)
    plane,support=support_plane(xyz,valid,seed);height=np.zeros_like(z)
    if plane is not None:height=np.clip(xyz@plane[0]+plane[1],-.5,.5)
    values=np.concatenate((z[None],height[None],normal.transpose(2,0,1),edge[None],valid[None].astype(np.float32)))
    masks=np.stack((valid,valid if plane is not None else np.zeros_like(valid),normal_valid,normal_valid,normal_valid,valid,np.ones_like(valid)))
    values=np.where(masks,values,0).astype(np.float32)
    assert np.isfinite(values).all()
    return values,masks,dict(plane_found=plane is not None,plane_support=support,valid_fraction=float(valid.mean()),normal_fraction=float(normal_valid.mean()))

def normalize(values,masks,stats):
    mean=np.asarray(stats['mean'],dtype=np.float32)[:,None,None]
    std=np.asarray(stats['std'],dtype=np.float32)[:,None,None]
    result=np.where(masks,(values-mean)/std,0).astype(np.float32)
    result[6]=values[6]
    return result
