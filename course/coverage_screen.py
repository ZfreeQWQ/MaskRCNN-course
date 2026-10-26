"""Train-only positive-coverage floors, isolated matcher or loss ablations.

Keep the official P-point budgets, cost/loss formulas and normalization. Sparse
GT-positive replacements introduce a deliberate sampling bias, NOT an unbiased
estimator. Matching still uses ONE coordinate set shared by all queries/GTs.
No RNG draws beyond the original point samplers. No inference architecture or
GT-driven inference proposals. Upstream files remain unchanged.
"""
import torch
from scipy.optimize import linear_sum_assignment
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness, point_sample,
)
from mask2former.modeling.matcher import HungarianMatcher, batch_dice_loss_jit, batch_sigmoid_ce_loss_jit
from mask2former.modeling.criterion import SetCriterion, calculate_uncertainty, dice_loss_jit, sigmoid_ce_loss_jit
from course.modeling_v2 import ROIMaskFormer


@torch.no_grad()
def positive_anchors(masks, small, points):
    """Deterministic foreground pixel-center quantiles, on the padded GT canvas.

    Sampling is per GT, never a union that would let large objects dominate.
    Pixel centers preserve a one-pixel target under align_corners=False. For
    fewer foreground pixels than anchors, repeated coordinates are intentional.
    This samples raster-order quantiles, not random/uniform foreground points.
    """
    if not isinstance(points, int) or points <= 0:
        raise ValueError("Positive integer anchor count required")
    if masks.ndim != 3 or small.shape != (len(masks),) or small.dtype != torch.bool:
        raise ValueError("Expected masks[N,H,W] and boolean small[N]")
    n,h,w = masks.shape
    result = torch.zeros(n, points, 2, device=masks.device, dtype=torch.float32)
    valid = torch.zeros(n, device=masks.device, dtype=torch.bool)
    for i in small.to(masks.device).nonzero().flatten():
        pixels = (masks[i] > .5).nonzero()
        if not len(pixels):
            continue
        ids = ((2*torch.arange(points, device=masks.device)+1)*len(pixels)) // (2*points)
        chosen = pixels[ids]
        result[i,:,0] = (chosen[:,1].float()+.5)/w
        result[i,:,1] = (chosen[:,0].float()+.5)/h
        valid[i] = True
    return result,valid


def shared_match_points(uniform, anchors, valid, cap):
    """Round-robin GT anchors replace the tail of a single common point set."""
    if uniform.ndim != 3 or uniform.shape[0] != 1 or uniform.shape[-1] != 2 or cap <= 0:
        raise ValueError("Expected common coords[1,P,2] and positive replacement cap")
    pool = anchors[valid].transpose(0,1).reshape(-1,2)
    k = min(len(pool), cap, uniform.shape[1])
    if not k:
        return uniform,0
    coords = uniform.clone()
    coords[:,-k:] = pool[:k]
    return coords,k


def matched_loss_points(uniform, anchors, valid):
    """Replace only the random tail for valid matched small GTs, not uncertainty slots."""
    if uniform.ndim != 3 or uniform.shape[0] != len(anchors) or valid.shape != (len(anchors),):
        raise ValueError("Matched anchors must align with per-instance coordinates")
    k = min(anchors.shape[1],uniform.shape[1])
    if not valid.any() or not k:
        return uniform,0
    coords = uniform.clone()
    coords[valid,-k:] = anchors[valid,:k]
    return coords,k*int(valid.sum())


class CoverageMatcher(HungarianMatcher):
    def __init__(self, *args, match_points=0, max_replacements=256, **kwargs):
        super().__init__(*args, **kwargs)
        if match_points < 0 or max_replacements <= 0:
            raise ValueError("Invalid coverage floor or cap")
        self.match_points,self.max_replacements=match_points,max_replacements
        self.last_diagnostics={}

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        if not self.match_points:
            return super().memory_efficient_forward(outputs,targets)
        result=[]
        replaced=small_targets=0
        for b,target in enumerate(targets):
            logits,masks=outputs['pred_logits'][b],outputs['pred_masks'][b]
            # Draw the full official uniform budget even on empty GT images.
            coords=torch.rand(1,self.num_points,2,device=masks.device)
            coords,k=shared_match_points(coords,target['coverage_points'][:,:self.match_points],
                                         target['coverage_valid'],self.max_replacements)
            replaced+=k;small_targets+=int(target['coverage_valid'].sum())
            if not len(target['labels']) or not len(masks):
                empty=torch.empty(0,dtype=torch.int64);result.append((empty,empty.clone()));continue
            class_cost=-logits.softmax(-1)[:,target['labels']]
            gt=point_sample(target['masks'].to(masks)[:,None],coords.repeat(len(target['masks']),1,1),align_corners=False).squeeze(1)
            pred=point_sample(masks[:,None],coords.repeat(len(masks),1,1),align_corners=False).squeeze(1)
            with torch.autocast(device_type=masks.device.type,enabled=False):
                cost=(self.cost_class*class_cost + self.cost_mask*batch_sigmoid_ce_loss_jit(pred.float(),gt.float())
                      + self.cost_dice*batch_dice_loss_jit(pred.float(),gt.float()))
            if not torch.isfinite(cost).all():
                raise ValueError('Nonfinite coverage matching cost')
            src,dst=linear_sum_assignment(cost.cpu())
            result.append((torch.as_tensor(src,dtype=torch.int64),torch.as_tensor(dst,dtype=torch.int64)))
        self.last_diagnostics=dict(replaced_points_across_images=replaced,valid_small_targets=small_targets,
                                   original_points_per_image=self.num_points,common_query_gt_coordinates=True)
        return result


class CoverageCriterion(SetCriterion):
    def __init__(self,*args,loss_points=0,**kwargs):
        super().__init__(*args,**kwargs)
        if loss_points < 0 or loss_points > self.num_points-int(self.num_points*self.importance_sample_ratio):
            raise ValueError('Floor must fit in the original random-point tail')
        self.loss_points=loss_points
        self.last_diagnostics={}

    def loss_masks(self,outputs,targets,indices,num_masks):
        if not self.loss_points:
            return super().loss_masks(outputs,targets,indices,num_masks)
        src=outputs['pred_masks'][self._get_src_permutation_idx(indices)][:,None]
        if not len(src):
            zero=outputs['pred_masks'].sum()*0
            return dict(loss_mask=zero,loss_dice=zero)
        # prepare_targets guarantees common padded canvas, already used upstream.
        if len({tuple(t['masks'].shape[-2:]) for t in targets})!=1:
            raise ValueError('Targets must share padded canvas')
        masks=torch.cat([t['masks'][j].to(src) for t,(_,j) in zip(targets,indices)])[:,None]
        anchors=torch.cat([t['coverage_points'][j,:self.loss_points].to(src) for t,(_,j) in zip(targets,indices)])
        valid=torch.cat([t['coverage_valid'][j].to(src.device) for t,(_,j) in zip(targets,indices)])
        with torch.no_grad():
            coords=get_uncertain_point_coords_with_randomness(src,calculate_uncertainty,self.num_points,
                                                              self.oversample_ratio,self.importance_sample_ratio)
            coords,k=matched_loss_points(coords,anchors,valid)
            labels=point_sample(masks,coords,align_corners=False).squeeze(1)
        pred=point_sample(src,coords,align_corners=False).squeeze(1)
        self.last_diagnostics=dict(replaced_points_across_matches=k,matched_small_instances=int(valid.sum()),
                                   original_points_per_instance=self.num_points)
        return dict(loss_mask=sigmoid_ce_loss_jit(pred,labels,num_masks),loss_dice=dice_loss_jit(pred,labels,num_masks))


@META_ARCH_REGISTRY.register()
class CoverageMaskFormer(ROIMaskFormer):
    @classmethod
    def from_config(cls,cfg):
        args=super().from_config(cfg)
        old=args['criterion'];m=old.matcher;c=cfg.COURSE.COVERAGE
        assert cfg.COURSE.V2.ROI_MATCH_WEIGHT==cfg.COURSE.V2.ROI_LOSS_FRACTION==0
        assert not cfg.COURSE.V2.AREA_WEIGHT
        matcher=CoverageMatcher(cost_class=m.cost_class,cost_mask=m.cost_mask,cost_dice=m.cost_dice,
                                num_points=m.num_points,match_points=c.MATCH_POINTS,
                                max_replacements=c.MAX_MATCH_REPLACEMENTS)
        args['criterion']=CoverageCriterion(old.num_classes,matcher,old.weight_dict,old.eos_coef,old.losses,
                                            old.num_points,old.oversample_ratio,old.importance_sample_ratio,
                                            loss_points=c.LOSS_POINTS)
        return args

    def prepare_targets(self,instances,images):
        targets=super().prepare_targets(instances,images)
        count=max(self.criterion.matcher.match_points,self.criterion.loss_points)
        if count:
            for target in targets:
                target['coverage_points'],target['coverage_valid']=positive_anchors(target['masks'],target['small'],count)
        return targets
