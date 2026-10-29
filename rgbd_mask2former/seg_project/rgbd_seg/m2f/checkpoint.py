"""Universal checkpointer for Mask2Former on this project (docs/plan.md §9, §11).

Handles two problems that plain DetectionCheckpointer gets wrong for our configs, both found
by GPU integration tests (not by static review -- see tools/tests/test_checkpoint_metadata.py
and the "size mismatch for ... query_feat.weight" crash this module's second half guards
against):

1. **Backbone key remap for I1** (only when the model actually uses the gated RGB-D backbone).
   rgbd_seg.m2f.depth_fusion.GatedRGBDResNetBackbone nests the ordinary (checkpoint-compatible)
   ResNet under a `rgb_backbone` submodule so it can coexist with the depth branch, which means
   its parameters show up as `backbone.rgb_backbone.stem...` instead of the plain
   `backbone.stem...` the Mask2Former COCO checkpoint uses. Without remapping,
   DetectionCheckpointer silently fails to match any backbone key and the whole RGB branch
   trains from random init -- silently defeating I1's entire point. This is detected at
   runtime (`hasattr(self.model.backbone, "rgb_backbone")`), NOT from a config flag, so the
   SAME checkpointer class is correct whether or not PROJECT.USE_DEPTH / I1 is enabled --
   applying the remap to a plain (non-gated) backbone would instead orphan every backbone key
   and silently randomize it, so this check matters. The remap itself is ALSO conditional per
   key (see `_remap_backbone`'s own docstring): it must not touch keys that are already correctly
   placed outside the plain ResNet (I1's own `depth_stem`/`depth_stages`/`fusions` layers, and
   an already-remapped `rgb_backbone.*`), which matters whenever the checkpoint being loaded is
   one of OUR OWN already-trained checkpoints rather than the original plain-ResNet COCO .pkl.

   IMPORTANT (found via tools/tests/test_depth_fusion_training_step.py's "checking that the
   pretrained RGB backbone actually loaded" step): the Mask2Former COCO .pkl checkpoint's
   `model` entry is an `OrderedDict` carrying a `_metadata` attribute (populated by the
   original `model.state_dict()` call used to build the zoo file, e.g.
   `{'sem_seg_head': {'version': 2}, ...}`). `MaskFormerHead._load_from_state_dict` reads this
   to know the checkpoint's pixel_decoder keys are already in the current naming scheme; if
   `_metadata` is missing, it assumes an OLD-format checkpoint and corrupts
   `sem_seg_head.pixel_decoder.*` into `sem_seg_head.pixel_decoder.pixel_decoder.*` -- silently
   dropping the entire pixel decoder (incl. the 6-layer deformable-attention encoder) back to
   random init. A naive dict rebuild loses `_metadata` because plain `dict` doesn't carry
   arbitrary attributes the way `OrderedDict` does; we rebuild as `OrderedDict` and explicitly
   remap+copy `_metadata` across.

2. **Query-embedding padding for NUM_OBJECT_QUERIES != 100.** The checkpoint's
   `sem_seg_head.predictor.static_query.weight` and `...query_embed.weight` are both
   (100, 256). Our configs use 200 queries (docs/plan.md's "up to 96 instances/image" headroom)
   so the target shape is (200, 256). `query_embed` keeps its name across checkpoint versions,
   so plain DetectionCheckpointer's built-in shape-mismatch check (in fvcore's
   `Checkpointer._load_model`, which runs BEFORE the nested module `_load_state_dict` calls)
   catches and cleanly drops it with a logged warning -- fine, if suboptimal (full reinit).
   `static_query`, however, is renamed to `query_feat` by
   `MultiScaleMaskedTransformerDecoder._load_from_state_dict`, a hook that only runs INSIDE
   `nn.Module.load_state_dict`'s recursive traversal, i.e. AFTER fvcore's shape pre-check has
   already passed it through unmodified (the pre-check never sees the post-rename name/shape).
   `nn.Module.load_state_dict(..., strict=False)` still hard-crashes on a shape mismatch
   regardless of `strict`, so this reliably raised `RuntimeError: size mismatch for
   sem_seg_head.predictor.query_feat.weight`. We fix both embeddings the same way here, before
   the rename hook ever runs: keep the pretrained weights for the first 100 query slots and
   fall back to the (randomly initialized) target model's own values for the rest -- strictly
   better than fvcore's drop-everything fallback for `query_embed`, and avoids the crash for
   `static_query`/`query_feat` entirely.
"""
from collections import OrderedDict

import torch
from detectron2.checkpoint import DetectionCheckpointer

# rgbd_seg.m2f.depth_fusion.GatedRGBDResNetBackbone's own top-level submodule names (as opposed
# to the plain ResNet's stem/res2../res5, which live INSIDE `rgb_backbone`). See
# _remap_backbone's docstring for why this whitelist exists.
_CUSTOM_BACKBONE_TOP_LEVEL = frozenset({"rgb_backbone", "depth_stem", "depth_stages", "fusions"})

_QUERY_KEYS = (
    # (checkpoint key name, target model key name after any submodule-internal rename)
    ("sem_seg_head.predictor.static_query.weight", "sem_seg_head.predictor.query_feat.weight"),
    ("sem_seg_head.predictor.query_embed.weight", "sem_seg_head.predictor.query_embed.weight"),
)


class M2FCheckpointer(DetectionCheckpointer):
    def _load_file(self, filename):
        checkpoint = super()._load_file(filename)
        model_dict = checkpoint["model"]

        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "rgb_backbone"):
            model_dict = self._remap_backbone(model_dict)

        model_dict = self._pad_query_embeddings(model_dict)

        checkpoint["model"] = model_dict
        return checkpoint

    @staticmethod
    def _remap_backbone(model_dict):
        """Remap backbone.X -> backbone.rgb_backbone.X, but ONLY for X belonging to a plain
        ResNet checkpoint (e.g. the original COCO .pkl's backbone.stem.*, backbone.res2.*, ...).

        FOUND VIA A REAL TEST-INFERENCE RUN producing garbage output (not by static review):
        the original version of this method remapped EVERY `backbone.*` key unconditionally,
        on the assumption that it only ever runs against the original plain-ResNet COCO
        checkpoint. That assumption breaks the moment you load one of OUR OWN already-trained
        checkpoints (e.g. resuming training, or running inference with best_segm_APs.pth):
        those checkpoints are saved FROM the multi-branch GatedRGBDResNetBackbone itself, so
        they already have the correct layout -- `backbone.rgb_backbone.*` (the plain ResNet,
        correctly nested), plus `backbone.depth_stem.*` / `backbone.depth_stages.*` /
        `backbone.fusions.*` (I1's depth branch and fusion layers, which are NOT nested under
        rgb_backbone -- see depth_fusion.py). Remapping unconditionally turned
        `backbone.depth_stem.*` into `backbone.rgb_backbone.depth_stem.*`, a path the model
        doesn't have, so fvcore's loader silently dropped it as "not found in the checkpoint"
        and left the model's REAL depth_stem/depth_stages/fusions parameters at their random
        init -- i.e. I1's entire trained depth-fusion contribution was discarded on every
        resume and every inference load, while the plain RGB backbone loaded "successfully"
        (producing no error, just wrong/degraded predictions).

        The fix: only remap a `backbone.X...` key when X's own top-level name is NOT one of
        the custom backbone's own submodule names (_CUSTOM_BACKBONE_TOP_LEVEL). This is
        correct for both checkpoint shapes without needing to detect which one is being
        loaded: the original COCO checkpoint has no `backbone.rgb_backbone.*` /
        `backbone.depth_*` / `backbone.fusions.*` keys at all (only backbone.stem/res2../res5),
        so every one of its backbone keys has top="stem"/"res2".."res5" and gets remapped, same
        as before. Our own checkpoints have ONLY keys already under one of the four custom
        top-level names, so none of them match and none get remapped.
        """
        remapped = OrderedDict()
        for key, value in model_dict.items():
            if key.startswith("backbone.") and not key.startswith("backbone.rgb_backbone."):
                rest = key[len("backbone."):]
                top = rest.split(".", 1)[0]
                if top not in _CUSTOM_BACKBONE_TOP_LEVEL:
                    remapped["backbone.rgb_backbone." + rest] = value
                    continue
            remapped[key] = value
        metadata = getattr(model_dict, "_metadata", None)
        if metadata is not None:
            # Same whitelist logic for module-path metadata keys (e.g. "backbone.stem" vs
            # "backbone.depth_stem"); the bare "backbone" key itself is left untouched in both
            # scenarios -- it is never consumed by any _load_from_state_dict hook in this
            # project (neither ResNet's nor GatedRGBDResNetBackbone's), so remapping it would
            # be a no-op at best and is skipped to avoid the same ambiguity that caused the bug
            # above.
            new_metadata = type(metadata)()
            for key, value in metadata.items():
                if key.startswith("backbone."):
                    rest = key[len("backbone."):]
                    top = rest.split(".", 1)[0]
                    if top not in _CUSTOM_BACKBONE_TOP_LEVEL:
                        key = "backbone.rgb_backbone." + rest
                new_metadata[key] = value
            remapped._metadata = new_metadata
        return remapped

    def _pad_query_embeddings(self, model_dict):
        target_state = self.model.state_dict()
        for ckpt_key, target_key in _QUERY_KEYS:
            if ckpt_key not in model_dict or target_key not in target_state:
                continue
            ckpt_tensor = model_dict[ckpt_key]
            ckpt_tensor = torch.as_tensor(ckpt_tensor) if not isinstance(ckpt_tensor, torch.Tensor) else ckpt_tensor
            target_tensor = target_state[target_key]
            if tuple(ckpt_tensor.shape) == tuple(target_tensor.shape):
                continue
            if ckpt_tensor.shape[1:] != target_tensor.shape[1:]:
                continue  # unexpected shape change beyond the query count; let the normal path handle/reject it
            old_n, new_n = ckpt_tensor.shape[0], target_tensor.shape[0]
            if new_n >= old_n:
                padded = target_tensor.clone()
                padded[:old_n] = ckpt_tensor
                model_dict[ckpt_key] = padded
            else:
                model_dict[ckpt_key] = ckpt_tensor[:new_n]
        return model_dict


# Backward-compatible alias: earlier plan iterations (and existing tests) named this class for
# its original, narrower purpose (backbone remap only, before query-padding was added).
RGBDBackboneCheckpointer = M2FCheckpointer
