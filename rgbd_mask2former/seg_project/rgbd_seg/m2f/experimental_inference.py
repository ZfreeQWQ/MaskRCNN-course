"""Controlled inference experiments; leave the established training/model code intact.

One backbone/decoder forward yields the legacy class-first and mask-first controls.
An optional DSH branch refines predicted small masks using the trained criterion head,
before binarization, with the same normalized/padded images used during training.
"""
import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.postprocessing import sem_seg_postprocess

from .inference import Predictor


def restore_logits(logits, padded_size, image_size, output_size):
    """Match MaskFormer's two-stage resize, including removal of image padding."""
    padded=F.interpolate(logits[None],size=padded_size,mode='bilinear',align_corners=False)[0]
    return sem_seg_postprocess(padded,image_size,*output_size)


def masks_and_quality(logits):
    masks=logits>0
    numeric=masks.to(logits.dtype)
    quality=(logits.sigmoid()*numeric).flatten(1).sum(1)/(numeric.flatten(1).sum(1)+1e-6)
    return masks,quality


def selection_indices(class_scores,quality,topk,ranking):
    if ranking not in ('class-first','mask-first'):
        raise ValueError('Unknown ranking')
    scores=class_scores if ranking=='class-first' else class_scores*quality
    # The legacy branch deliberately matches upstream topk(sorted=False).
    return scores.topk(min(topk,len(scores)),sorted=False).indices


def make_instances(masks,scores,indices):
    result=Instances(masks.shape[-2:])
    result.pred_masks=masks[indices]
    result.scores=scores[indices]
    result.pred_classes=torch.zeros(len(indices),dtype=torch.int64,device=masks.device)
    result.pred_boxes=Boxes(torch.zeros((len(indices),4),device=masks.device))
    return result


class ExperimentalPredictor(Predictor):
    def __init__(self,cfg,weights,refine_small=False,chunk_size=8):
        super().__init__(cfg,weights)
        if self.model.training or self.model.sem_seg_head.num_classes!=1:
            raise ValueError('This experiment supports eval-mode single-class instance segmentation')
        if not self.model.sem_seg_postprocess_before_inference:
            raise ValueError('Expected original-size instance logits before postprocessing')
        if chunk_size<1:
            raise ValueError('chunk_size must be positive')
        self.refine_small=refine_small
        self.chunk_size=chunk_size
        self.head=getattr(self.model.criterion,'hires_head',None)
        if refine_small and self.head is None:
            raise ValueError('Refinement requested but checkpoint model has no hires head')
        self.head_residual_norm=float(self.head.residual[-1].weight.norm().item()) if self.head is not None else None
        if refine_small and self.head_residual_norm==0:
            raise ValueError('The loaded refinement head is still zero-initialized')

    @torch.inference_mode()
    def __call__(self,record,check_reference=False):
        model=self.model
        mapped=self.mapper(record)
        image=mapped['image'].to(model.device)
        normalized=(image-model.pixel_mean)/model.pixel_std
        images=ImageList.from_tensors([normalized],model.size_divisibility)
        output=model.sem_seg_head(model.backbone(images.tensor))
        coarse=output['pred_masks'][0]
        class_logits=output['pred_logits'][0]
        del output
        classes=class_logits.softmax(-1)[:,0]
        padded_size=images.tensor.shape[-2:]
        image_size=images.image_sizes[0]
        output_size=(record['height'],record['width'])
        n=len(classes)
        masks=torch.empty((n,*output_size),dtype=torch.bool,device=coarse.device)
        quality=torch.empty_like(classes)
        for start in range(0,n,self.chunk_size):
            end=min(start+self.chunk_size,n)
            logits=restore_logits(coarse[start:end],padded_size,image_size,output_size)
            masks[start:end],quality[start:end]=masks_and_quality(logits)
        scores=classes*quality
        legacy_ids=selection_indices(classes,quality,model.test_topk_per_image,'class-first')
        reranked_ids=selection_indices(classes,quality,model.test_topk_per_image,'mask-first')
        variants={'legacy':make_instances(masks,scores,legacy_ids),
                  'mask_first':make_instances(masks,scores,reranked_ids)}
        diagnostics=dict(num_queries=n,selection_replacements=len(set(reranked_ids.tolist())-set(legacy_ids.tolist())),
                         refined_queries=0,refined_changed_pixels=0,head_residual_norm=self.head_residual_norm)
        if check_reference:
            restored=restore_logits(coarse,padded_size,image_size,output_size)
            reference=model.instance_inference(class_logits,restored)
            if not torch.equal(reference.pred_masks.bool(),variants['legacy'].pred_masks):
                raise AssertionError('Legacy masks differ from upstream instance_inference')
            torch.testing.assert_close(reference.scores,variants['legacy'].scores,rtol=1e-6,atol=1e-7)
            diagnostics['legacy_matches_upstream']=True
            del restored,reference
        if self.refine_small:
            selected_masks=masks[legacy_ids].clone()
            selected_scores=scores[legacy_ids].clone()
            # This gate depends only on predictions in original output coordinates.
            areas=selected_masks.flatten(1).sum(1)
            positions=((areas>0)&(areas<=1024)).nonzero(as_tuple=True)[0]
            diagnostics['refined_queries']=len(positions)
            if len(positions):
                image_features=self.head.image_features(images.tensor)
                for start in range(0,len(positions),self.chunk_size):
                    loc=positions[start:start+self.chunk_size]
                    ids=legacy_ids[loc]
                    batch_idx=torch.zeros(len(ids),dtype=torch.int64,device=coarse.device)
                    refined=self.head.refine(coarse[ids],image_features,batch_idx)
                    logits=restore_logits(refined,padded_size,image_size,output_size)
                    new_masks,new_quality=masks_and_quality(logits)
                    diagnostics['refined_changed_pixels']+=int((new_masks!=selected_masks[loc]).sum().item())
                    selected_masks[loc]=new_masks
                    selected_scores[loc]=classes[ids]*new_quality
                del image_features
            variants['refined_small']=make_instances(selected_masks,selected_scores,
                                                     torch.arange(len(legacy_ids),device=coarse.device))
        return variants,diagnostics
