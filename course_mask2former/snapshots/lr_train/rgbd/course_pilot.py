"""Training-only GT-piloted auxiliary decoder, inspired by MP-Former.

Own bounded no-noise variant, not an exact reproduction. Ordinary queries,
Hungarian losses and inference remain unchanged. No GT is used by predict.
"""
import torch
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.registry import MODELS
from rgbd.multiscale import MultiScaleRGBDMask2Former
from m2f_local_decoder import guided_loss

def selected_indices(n,step,image_index,device):
    if n==0:return torch.empty(0,dtype=torch.long,device=device)
    # Fixed rotation, no extra global RNG consumption and no area threshold.
    return (torch.arange(min(n,16),device=device)+((step+image_index*7)%n))%n

class PilotedHead(Mask2FormerHead):
    def loss(self,x,batch_data_samples):
        assert self.training
        captured=[]
        h=self.pixel_decoder.register_forward_hook(lambda module,args,out:captured.append(out))
        try:losses=super().loss(x,batch_data_samples)
        finally:h.remove()
        assert len(captured)==1,'Only one shared pixel decoder forward per train batch'
        pixels,memories=captured[0]
        gt=self.preprocess_gt([d.gt_instances for d in batch_data_samples],
            [d.gt_sem_seg if 'gt_sem_seg' in d else None for d in batch_data_samples])
        aux=pixels.sum()*0;count=0
        devices=[pixels.device.index] if pixels.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            # The auxiliary random point sampling cannot perturb the ordinary
            # next-step sampler/model RNG. No newly randomized parameters.
            torch.manual_seed(20261013+self.pilot_calls)
            for i,target in enumerate(gt):
                indices=selected_indices(len(target.labels),self.pilot_calls,i,pixels.device)
                if not len(indices):continue
                selected=InstanceData(labels=target.labels[indices],masks=target.masks[indices])
                assert selected.masks.flatten(1).any(1).all()
                aux=aux+guided_loss(self,pixels[i:i+1],[m[i:i+1] for m in memories],selected)
                count+=len(indices)
        losses['loss_mask_pilot']=aux*(.5/len(gt))
        self.pilot_calls+=1;self.pilot_samples+=len(gt);self.pilot_queries+=count
        return losses

@MODELS.register_module()
class PilotedRGBDMask2Former(MultiScaleRGBDMask2Former):
    def __init__(self,*args,course_pilot=None,**kwargs):
        options=dict(course_pilot or {})
        if options!=dict(enabled=True,max_queries=16,weight=.5,no_noise=True):
            raise ValueError('Frozen pilot treatment only')
        super().__init__(*args,**kwargs)
        assert isinstance(self.panoptic_head,Mask2FormerHead)
        self.panoptic_head.__class__=PilotedHead
        self.panoptic_head.pilot_calls=0
        self.panoptic_head.pilot_samples=0
        self.panoptic_head.pilot_queries=0
