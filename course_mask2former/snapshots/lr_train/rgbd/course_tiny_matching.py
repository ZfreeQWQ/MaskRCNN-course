"""Fixed tiny foreground quota only for shared Hungarian matching coordinates."""
import torch
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from rgbd.course_pointnoise import NoisyPilotedHead,NoisyPilotedRGBDMask2Former

def tiny_pixels(masks):
    positive=masks>.5;areas=positive.flatten(1).sum(1)
    indices=((areas>0)&(areas<=64)).nonzero().flatten()
    return [(int(i),positive[i].flatten().nonzero().flatten()) for i in indices],masks.shape[-2:]

def replace_tail(coords,pixels,shape,generator,quota=1024):
    assert coords.shape==(1,12544,2) and quota==1024
    out=coords.clone()
    if not pixels:return out,0
    h,w=shape;n=len(pixels);assert n<=quota
    for j,(_,sites) in enumerate(pixels):
        destinations=torch.arange(j,quota,n,device=coords.device)
        chosen=sites[torch.randint(sites.numel(),(len(destinations),),device=coords.device,generator=generator)]
        centres=torch.stack(((chosen%w).float()+.5,torch.div(chosen,w,rounding_mode='floor').float()+.5),-1)
        out[0,self_start(coords,quota)+destinations]=centres/centres.new_tensor([w,h])
    return out,quota

def self_start(coords,quota):return coords.shape[1]-quota

class TinyMatchedHead(NoisyPilotedHead):
    def loss(self,*args,**kwargs):
        self._tiny_cache={}
        try:return super().loss(*args,**kwargs)
        finally:self._tiny_cache={}

    def _get_targets_single(self,cls_score,mask_pred,gt_instances,img_meta):
        gt_labels=gt_instances.labels;gt_masks=gt_instances.masks
        num_queries=cls_score.shape[0];num_gts=gt_labels.shape[0]
        # Same global draw as the untouched implementation, before any replacement.
        point_coords=torch.rand((1,self.num_points,2),device=cls_score.device)
        if self.tiny_matching_enabled:
            assert self.training and self.num_points==12544
            key=(gt_masks.data_ptr(),tuple(gt_masks.shape))
            cache=getattr(self,'_tiny_cache',{})
            if key not in cache:cache[key]=tiny_pixels(gt_masks)
            self._tiny_cache=cache;pixels,shape=cache[key]
            g=torch.Generator(device=cls_score.device).manual_seed(731527+104729*self.match_activity['calls'])
            point_coords,replaced=replace_tail(point_coords,pixels,shape,g)
            self.match_activity['calls']+=1;self.match_activity['tiny_targets']+=len(pixels);self.match_activity['replaced_points']+=replaced
        mask_points_pred=point_sample(mask_pred.unsqueeze(1),point_coords.repeat(num_queries,1,1)).squeeze(1)
        gt_points_masks=point_sample(gt_masks.unsqueeze(1).float(),point_coords.repeat(num_gts,1,1)).squeeze(1)
        sampled_gt_instances=InstanceData(labels=gt_labels,masks=gt_points_masks)
        sampled_pred_instances=InstanceData(scores=cls_score,masks=mask_points_pred)
        assign_result=self.assigner.assign(pred_instances=sampled_pred_instances,gt_instances=sampled_gt_instances,img_meta=img_meta)
        pred_instances=InstanceData(scores=cls_score,masks=mask_pred)
        sampling_result=self.sampler.sample(assign_result=assign_result,pred_instances=pred_instances,gt_instances=gt_instances)
        pos_inds=sampling_result.pos_inds;neg_inds=sampling_result.neg_inds
        labels=gt_labels.new_full((self.num_queries,),self.num_classes,dtype=torch.long)
        labels[pos_inds]=gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights=gt_labels.new_ones((self.num_queries,))
        mask_targets=gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights=mask_pred.new_zeros((self.num_queries,));mask_weights[pos_inds]=1.0
        return labels,label_weights,mask_targets,mask_weights,pos_inds,neg_inds,sampling_result

@MODELS.register_module()
class TinyMatchedRGBDMask2Former(NoisyPilotedRGBDMask2Former):
    def __init__(self,*args,tiny_matching=None,**kwargs):
        assert tiny_matching==dict(enabled=True,quota=1024,area_threshold=64)
        super().__init__(*args,**kwargs)
        self.panoptic_head.__class__=TinyMatchedHead
        self.panoptic_head.tiny_matching_enabled=True
        self.panoptic_head.match_activity=dict(calls=0,tiny_targets=0,replaced_points=0)
        self.panoptic_head._tiny_cache={}
