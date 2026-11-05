"""Label-free quality descriptors using a learned RGB-D adjacency field."""
import numpy as np
from pycocotools import mask as mu

NAMES=('score','score_logit','log_area','log_width','log_height','fill','log_aspect','perimeter_scaled','boundary_mean','boundary_std','boundary_q25','boundary_q50','boundary_q75','inside_mean','inside_std','inside_q10','inside_q50','inside_boundary_margin','strong_boundary_fraction','valid_foreground','valid_boundary_pairs','mean_depth_m','depth_std_2mm')

def describe(pred,wx,wy,z,valid):
    area=float(mu.area(pred['segmentation']));bx,by,bw,bh=mu.toBbox(pred['segmentation']);x=max(0,int(bx)-1);y=max(0,int(by)-1);x2=min(1024,int(bx+bw)+1);y2=min(1024,int(by+bh)+1)
    m=mu.decode(pred['segmentation'])[y:y2,x:x2].astype(bool);vx=valid[y:y2,x:x2-1]&valid[y:y2,x+1:x2];vy=valid[y:y2-1,x:x2]&valid[y+1:y2,x:x2]
    cx=m[:,:-1]!=m[:,1:];cy=m[:-1,:]!=m[1:,:];ix=m[:,:-1]&m[:,1:];iy=m[:-1,:]&m[1:,:]
    ax=wx[y:y2,x:x2-1];ay=wy[y:y2-1,x:x2]
    boundary=np.concatenate([ax[cx],ay[cy]]);inside=np.concatenate([ax[ix],ay[iy]])
    bmean=float(boundary.mean()) if len(boundary) else 0.;imean=float(inside.mean()) if len(inside) else 0.
    bq=np.quantile(boundary,[.25,.5,.75]).tolist() if len(boundary) else [0.,0.,0.];iq=np.quantile(inside,[.1,.5]).tolist() if len(inside) else [0.,0.]
    vfg=valid[y:y2,x:x2][m];depth=z[y:y2,x:x2][m&valid[y:y2,x:x2]];score=float(pred['score'])
    result=[score,float(np.log(np.clip(score,1e-5,1-1e-5)/np.clip(1-score,1e-5,1-1e-5)))/10,np.log1p(area)/8,np.log1p(bw)/7,np.log1p(bh)/7,area/max(1,bw*bh),abs(np.log(max(bw,1)/max(bh,1)))/4,len(boundary)/(4*np.sqrt(max(area,1))),bmean,float(boundary.std()) if len(boundary) else 0.,*bq,imean,float(inside.std()) if len(inside) else 0.,*iq,imean-bmean,float((boundary<.1).mean()) if len(boundary) else 0.,float(vfg.mean()) if len(vfg) else 0.,float(np.r_[vx[cx],vy[cy]].mean()) if len(boundary) else 0.,float(depth.mean()) if len(depth) else 0.,float(depth.std()/.002) if len(depth) else 0.]
    assert len(result)==len(NAMES)
    return np.nan_to_num(np.asarray(result,np.float32),nan=0,posinf=10,neginf=-10).clip(-10,10)
