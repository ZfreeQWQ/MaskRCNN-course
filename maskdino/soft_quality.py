"""Frozen MaskDINO soft-logit descriptors and current best native mask pipeline."""
import numpy as np
import torch as t
import types,copy
F=t.nn.functional
import refine_inference_20261103 as h
from soft_runtime import build,native,load_parent_head
from quality_refinement_core import refine_and_weights
from tiny_seed_boundary import refine_skipped
from native_quality_features import describe,NAMES

SOFT_NAMES=('class_probability','raw_mask_probability_mean','raw_mask_probability_std','raw_mask_probability_q10','raw_mask_probability_q25','raw_mask_probability_q50','raw_mask_probability_q75','raw_mask_probability_q90','raw_positive_entropy','raw_inner_boundary_probability','raw_inner_boundary_entropy','raw_outer_boundary_probability','raw_outer_boundary_q90','raw_outer_boundary_entropy','raw_logit_abs_q10','raw_logit_abs_mean','raw_soft_area_ratio','raw_boundary_uncertain_fraction','native_view_side_1024','internal_border_clearance_64')
FEATURES=NAMES+SOFT_NAMES

class SoftCapture:
    def __init__(self,model):
        old=model.instance_inference;self.logits=None;self.probabilities=None
        def wrapped(m,cls,masks,boxes):
            result=old(cls,masks,boxes);assert cls.shape[-1]==1
            probs=cls.sigmoid().flatten();ids=probs.topk(m.test_topk_per_image,sorted=False).indices
            chosen=masks[ids];assert t.equal(chosen>0,result.pred_masks.bool())
            # Float32 keeps the exact zero boundary used by the original masks.
            self.logits=chosen.detach().float().cpu();self.probabilities=probs[ids].detach().float().cpu()
            return result
        model.instance_inference=types.MethodType(wrapped,model)

def soft_features(logit,mask,class_prob,side,clearance):
    yy,xx=np.nonzero(mask);x=max(0,int(xx.min())-2);y=max(0,int(yy.min())-2);x2=min(side,int(xx.max())+3);y2=min(side,int(yy.max())+3)
    m=mask[y:y2,x:x2];l=logit[y:y2,x:x2].numpy();prob=1/(1+np.exp(-l.clip(-20,20)))
    mt=t.from_numpy(m.astype(np.float32))[None,None];inner=(1-F.max_pool2d(1-mt,3,1,1))[0,0].numpy().astype(bool);outer=F.max_pool2d(mt,3,1,1)[0,0].numpy().astype(bool)
    inside=m&~inner;outside=outer&~m;positive=prob[m]
    entropy=-(prob*np.log(prob.clip(1e-7,1))+(1-prob)*np.log((1-prob).clip(1e-7,1)))/np.log(2)
    def mean(values):return float(values.mean()) if len(values) else 0.
    outer_values=prob[outside];boundary=inside|outside
    values=[class_prob,mean(positive),float(positive.std()),*np.quantile(positive,[.1,.25,.5,.75,.9]).tolist(),mean(entropy[m]),mean(prob[inside]),mean(entropy[inside]),mean(outer_values),float(np.quantile(outer_values,.9)) if len(outer_values) else 0.,mean(entropy[outside]),float(np.quantile(np.abs(l[m]),.1))/10,mean(np.abs(l[m]))/10,float(prob.sum()/max(1,m.sum())),mean(((prob[boundary]>=.4)&(prob[boundary]<=.6)).astype(np.float32)),side/1024.,min(4.,clearance/64.)]
    assert len(values)==len(SOFT_NAMES)
    return np.nan_to_num(np.asarray(values,np.float32),nan=0,posinf=10,neginf=0).clip(0,10)

@t.inference_mode()
def raw_proposals(model,capture,data,i):
    item=data.item(i,False);image=item['image'];result=[];soft={}
    for x,y,side in [(0,0,1024),(0,0,640),(384,0,640),(0,384,640),(384,384,640)]:
        z=image[:,y:y+side,x:x+side]
        z=t.cat((F.interpolate(z[None,:3],(2048,2048),mode='bilinear',align_corners=False)[0],F.interpolate(z[None,3:],(2048,2048),mode='nearest')[0]))
        pred=model([dict(image=z,height=side,width=side)])[0]['instances'].to('cpu');logits=capture.logits;classes=capture.probabilities
        capture.logits=None;capture.probabilities=None
        for j,(mask,score) in enumerate(zip(pred.pred_masks,pred.scores)):
            m=mask.bool().numpy();area=int(m.sum())
            if not area or (side!=1024 and area>2048):continue
            yy,xx=np.nonzero(m);edges=[]
            if x>0:edges.append(int(xx.min()))
            if y>0:edges.append(int(yy.min()))
            if x+side<1024:edges.append(side-1-int(xx.max()))
            if y+side<1024:edges.append(side-1-int(yy.max()))
            clearance=min(edges) if edges else 256
            if side!=1024 and clearance<8:continue
            full=np.zeros((1024,1024),np.uint8);full[y:y+side,x:x+side]=m
            key=len(result);p=dict(image_id=item['image_id'],category_id=1,score=float(score),segmentation=h.enc(full),_soft_key=key)
            if area<=2048:
                values=soft_features(logits[j],m,float(classes[j]),side,clearance)
                assert abs(float(score)-float(classes[j])*float(values[1]))<1e-5,'Captured class/mask confidence alignment mismatch'
                soft[key]=values
            result.append(p)
        del logits,classes,pred
    return h.merge(result,[]),soft

def clean(p):return {key:p[key] for key in ['image_id','category_id','score','segmentation']}

@t.inference_mode()
def current_features(model,capture,data,i):
    raw,soft=raw_proposals(model,capture,data,i);rgb,z,valid,k=native(data,i)
    parent,wx,wy=refine_and_weights(raw,rgb,z,valid,k);updated,_=refine_skipped(parent,raw,wx,wy);ax=wx[0,0].cpu().numpy();ay=wy[0,0].cpu().numpy()
    ids=[j for j,p in enumerate(updated) if 0<float(h.mu.area(p['segmentation']))<=2048]
    feature_map={}
    if ids:
        core=np.asarray([describe(updated[j],ax,ay,z,valid) for j in ids],np.float32);head,mean,std=load_parent_head();quality=head(t.from_numpy((core-mean)/std).cuda()).mean(1).cpu().numpy()
        for row,j,q in zip(core,ids,quality):
            p=updated[j];feature_map[p['_soft_key']]=np.r_[row,soft[p['_soft_key']]].astype(np.float32);p['_raw_score']=p['score'];p['score']*=float(q)
    # Freeze the successfully validated parent's mask set and NMS selection.
    selected=h.merge(updated,[]);baseline=[clean(p) for p in selected];features=[];positions=[]
    for j,p in enumerate(selected):
        if p['_soft_key'] in feature_map:
            features.append(feature_map[p['_soft_key']]);positions.append(j);p['score']=p['_raw_score']
    return selected,np.asarray(features,np.float32).reshape(-1,len(FEATURES)),positions,baseline,[clean(p) for p in raw]
