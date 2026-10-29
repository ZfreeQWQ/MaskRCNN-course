"""Area-capped attention point noise; clean targets and ordinary path unchanged.

MP-Former-inspired bounded variant, not a reproduction. Noise is applied on
each attention grid, never to the loss target, never during inference. Very
small guides (<5 occupied cells) have a zero integer noise budget.
"""
import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from rgbd.course_pilot import PilotedHead,PilotedRGBDMask2Former,selected_indices

def noisy_occupancy(allowed,generator,ratio=.2):
    """Uniform independent counts 0..floor(ratio*A), unique uniform flip sites."""
    if not 0<=ratio<=.2:raise ValueError('No noise-strength search')
    assert allowed.dtype==torch.bool and allowed.ndim==3
    flat=allowed.flatten(1);areas=flat.sum(1)
    assert bool((areas>0).all())
    limits=torch.floor(areas.double()*ratio).long()
    counts=torch.floor(torch.rand(len(flat),device=flat.device,generator=generator)*(limits+1)).long()
    maximum=int(counts.max()) if len(counts) else 0
    result=flat.clone()
    if maximum:
        # Random-key sampling without replacement, one batched draw per layer.
        keys=torch.rand(flat.shape,device=flat.device,generator=generator)
        sites=keys.topk(maximum,dim=1,largest=False).indices
        flips=torch.zeros_like(flat)
        flips.scatter_(1,sites,torch.arange(maximum,device=flat.device)[None]<counts[:,None])
        result^=flips
    assert bool(result.any(1).all()),'Area cap must preserve foreground'
    return result.reshape_as(allowed),counts,areas

def decoder(head,pixels,memories,labels,targets,generator,ratio,activity):
    batch=pixels.shape[0];assert batch==1
    inputs=[];positions=[]
    for i in range(head.num_transformer_feat_level):
        z=head.decoder_input_projs[i](memories[i]).flatten(2).permute(0,2,1)
        inputs.append(z+head.level_embed.weight[i].view(1,1,-1))
        empty=z.new_zeros((batch,)+memories[i].shape[-2:],dtype=torch.bool)
        positions.append(head.decoder_positional_encoding(empty).flatten(2).permute(0,2,1))
    assert targets is not None and len(labels)==len(targets) and len(labels)>0
    query=head.cls_embed.weight.detach()[labels][None].clone()
    position=torch.zeros_like(query)
    cs=[];ms=[]
    c,m,attention=head._forward_head(query,pixels,memories[0].shape[-2:]);cs.append(c);ms.append(m)
    for i,layer in enumerate(head.transformer_decoder.layers):
        level=i%head.num_transformer_feat_level
        clean=F.adaptive_max_pool2d(targets[:,None].float(),memories[level].shape[-2:])[:,0]>0
        allowed,counts,areas=noisy_occupancy(clean,generator,ratio)
        activity.add_(torch.stack((counts.sum(),areas.sum(),(counts>0).sum(),(areas<5).sum(),areas.new_tensor(len(areas)))))
        attention=(~allowed).flatten(1)[None].repeat(head.num_heads,1,1).detach()
        query=layer(query=query,key=inputs[level],value=inputs[level],query_pos=position,
            key_pos=positions[level],cross_attn_mask=attention,query_key_padding_mask=None,key_padding_mask=None)
        c,m,attention=head._forward_head(query,pixels,memories[(i+1)%head.num_transformer_feat_level].shape[-2:])
        cs.append(c);ms.append(m)
    return cs,ms

def guided_loss(head,pixels,memories,gt,generator,ratio,activity):
    n=len(gt.labels)
    if not n:return sum(p.sum()*0 for p in head.parameters() if p.requires_grad)
    cs,ms=decoder(head,pixels,memories,gt.labels,gt.masks,generator,ratio,activity)
    total=pixels.new_zeros(())
    for c,m in zip(cs,ms):
        total=total+head.loss_cls(c[0],gt.labels,avg_factor=n)
        pred=m[0]
        with torch.no_grad():
            coords=get_uncertain_point_coords_with_randomness(pred[:,None],None,head.num_points,head.oversample_ratio,head.importance_sample_ratio)
            target=point_sample(gt.masks[:,None].float(),coords).squeeze(1)
        pred=point_sample(pred[:,None],coords).squeeze(1)
        total=total+head.loss_dice(pred,target,avg_factor=n)+head.loss_mask(pred.reshape(-1),target.reshape(-1),avg_factor=n*head.num_points)
    return total

class NoisyPilotedHead(PilotedHead):
    def loss(self,x,batch_data_samples):
        assert self.training
        captured=[]
        h=self.pixel_decoder.register_forward_hook(lambda module,args,out:captured.append(out))
        try:losses=Mask2FormerHead.loss(self,x,batch_data_samples)
        finally:h.remove()
        assert len(captured)==1
        pixels,memories=captured[0]
        gt=self.preprocess_gt([d.gt_instances for d in batch_data_samples],
            [d.gt_sem_seg if 'gt_sem_seg' in d else None for d in batch_data_samples])
        aux=pixels.sum()*0;count=0
        activity=torch.zeros(5,device=pixels.device,dtype=torch.long)
        devices=[pixels.device.index] if pixels.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261013+self.pilot_calls)
            for i,target in enumerate(gt):
                indices=selected_indices(len(target.labels),self.pilot_calls,i,pixels.device)
                if not len(indices):continue
                selected=InstanceData(labels=target.labels[indices],masks=target.masks[indices])
                assert selected.masks.flatten(1).any(1).all()
                # Dedicated RNG leaves ordinary AND auxiliary point-sampling RNG intact.
                generator=torch.Generator(device=pixels.device).manual_seed(917361+104729*self.pilot_calls+i)
                aux=aux+guided_loss(self,pixels[i:i+1],[m[i:i+1] for m in memories],selected,
                    generator,self.point_noise_ratio,activity)
                count+=len(indices)
        losses['loss_mask_pilot']=aux*(.5/len(gt))
        self.pilot_calls+=1;self.pilot_samples+=len(gt);self.pilot_queries+=count
        for key,value in zip(('flips','guide_foreground','noised_guides','tiny_unchanged_guides','guides'),activity.tolist()):
            self.noise_activity[key]+=value
        self.noise_activity['train_calls']+=1
        return losses

@MODELS.register_module()
class NoisyPilotedRGBDMask2Former(PilotedRGBDMask2Former):
    def __init__(self,*args,course_pilot=None,**kwargs):
        if dict(course_pilot or {})!=dict(enabled=True,max_queries=16,weight=.5,no_noise=False,point_noise_ratio=.2):
            raise ValueError('One fixed area-capped point-noise treatment only')
        super().__init__(*args,course_pilot=dict(enabled=True,max_queries=16,weight=.5,no_noise=True),**kwargs)
        self.panoptic_head.__class__=NoisyPilotedHead
        self.panoptic_head.point_noise_ratio=.2
        self.panoptic_head.noise_activity={k:0 for k in ('train_calls','flips','guide_foreground','noised_guides','tiny_unchanged_guides','guides')}
