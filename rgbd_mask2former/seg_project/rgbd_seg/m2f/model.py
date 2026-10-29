"""RGBDMaskFormer: Mask2Former variant wiring in I2 (small-object matcher/criterion) and I3
(depth-guided hi-res mask refinement) -- docs/plan.md §4 I2, I3. Registered under
META_ARCHITECTURE "RGBDMaskFormer" so configs opt in explicitly; the official "MaskFormer" is
left importable and usable unchanged (used as the plain baseline B).

I1 (depth fusion) is independent of this class -- it's selected purely via
MODEL.BACKBONE.NAME=build_gated_rgbd_resnet_backbone (rgbd_seg.m2f.depth_fusion), which any
meta-architecture can use.

When PROJECT.SMALL_OBJECT_POINTS.ENABLED is off and PROJECT.HIRES_MASK.ENABLED is off, this
class is behaviorally identical to `MaskFormer` (inherits forward() unmodified; from_config
returns the base dict as-is), so e.g. the "B" experiment in docs/plan.md §5 can use this same
class name throughout without a behavior change, avoiding a from-scratch-equal check between two
different registered classes.
"""
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from mask2former.maskformer_model import MaskFormer

from .matcher import SmallObjectMatcher
from .criterion import SmallObjectCriterion, HiResCriterion
from .hires_mask import HiResMaskHead


@META_ARCH_REGISTRY.register()
class RGBDMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        sop = cfg.PROJECT.SMALL_OBJECT_POINTS
        hires = cfg.PROJECT.HIRES_MASK
        if not sop.ENABLED and not hires.ENABLED:
            return ret

        old_criterion = ret["criterion"]
        base_matcher = old_criterion.matcher
        if sop.ENABLED:
            matcher = SmallObjectMatcher(
                cost_class=base_matcher.cost_class,
                cost_mask=base_matcher.cost_mask,
                cost_dice=base_matcher.cost_dice,
                num_points=base_matcher.num_points,
                extra_points=sop.NUM_POINTS,
                box_expand=sop.BOX_EXPAND,
            )
        else:
            matcher = base_matcher

        criterion_cls = HiResCriterion if hires.ENABLED else SmallObjectCriterion
        kwargs = dict(
            num_classes=old_criterion.num_classes,
            matcher=matcher,
            weight_dict=old_criterion.weight_dict,
            eos_coef=old_criterion.eos_coef,
            losses=old_criterion.losses,
            num_points=old_criterion.num_points,
            oversample_ratio=old_criterion.oversample_ratio,
            importance_sample_ratio=old_criterion.importance_sample_ratio,
            loss_point_fraction=sop.LOSS_POINT_FRACTION if sop.ENABLED else 0.0,
            box_expand=sop.BOX_EXPAND,
            area_weight_enabled=sop.AREA_WEIGHT_ENABLED,
            area_weight_ref=sop.AREA_WEIGHT_REF,
            area_weight_power=sop.AREA_WEIGHT_POWER,
            area_weight_max=sop.AREA_WEIGHT_MAX,
        )
        if hires.ENABLED:
            in_channels = 9 if cfg.PROJECT.USE_DEPTH and hires.USE_DEPTH else 3
            hires_head = HiResMaskHead(in_channels=in_channels)
            kwargs["hires_head"] = hires_head
            kwargs["hires_max_pairs"] = hires.MAX_PAIRS_PER_STEP
            # The hires branch only runs on the main decoder output, not the aux (deep
            # supervision) layers (see HiResCriterion's module docstring for why -- an earlier
            # version that ran it on all 10 layers OOMed), so it only ever produces the two
            # un-suffixed keys below, not loss_mask_hires_0..8. Give them the same weight as
            # their coarse counterparts; otherwise MaskFormer.forward()'s "remove this loss if
            # not specified in weight_dict" would silently drop them (see third_party/
            # Mask2Former/mask2former/maskformer_model.py's forward()).
            weight_dict = dict(old_criterion.weight_dict)
            weight_dict["loss_mask_hires"] = weight_dict["loss_mask"]
            weight_dict["loss_dice_hires"] = weight_dict["loss_dice"]
            kwargs["weight_dict"] = weight_dict
            ret["criterion"] = criterion_cls(**kwargs)
            ret["criterion"].weight_dict = weight_dict  # also reflected on the instance itself
        else:
            ret["criterion"] = criterion_cls(**kwargs)
        return ret

    def forward(self, batched_inputs):
        # I3 (hires_head) needs the model's own preprocessed input tensor inside the criterion,
        # which plain MaskFormer.forward() never exposes (it only passes `outputs`/`targets`).
        # Eval-mode behavior is unchanged, so we only override the training path; everything
        # here through `images = ImageList.from_tensors(...)` is copied verbatim from
        # MaskFormer.forward (third_party/Mask2Former/mask2former/maskformer_model.py) -- keep
        # the two in sync if that method's preprocessing ever changes.
        if not self.training:
            return super().forward(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances, images)

        if isinstance(self.criterion, HiResCriterion):
            # Only HiResCriterion.forward accepts `images=`; plain SetCriterion/SmallObjectCriterion
            # inherit SetCriterion.forward(self, outputs, targets) with no **kwargs, so passing
            # images= to them would raise a TypeError.
            losses = self.criterion(outputs, targets, images=images.tensor)
        else:
            losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses
