"""Paired, prediction-only mask-first inference for the independent D+H ablation.

Inputs are already mapped by the validation loader. Both branches use the same
100 query IDs, selected before refinement; GT never enters the refinement gate.
"""
import torch
from torch import nn
from detectron2.structures import ImageList

from .experimental_inference import (
    restore_logits, masks_and_quality, selection_indices, make_instances,
)


@torch.inference_mode()
def paired_prediction(model, record, chunk_size=4):
    if chunk_size < 1 or model.sem_seg_head.num_classes != 1:
        raise ValueError("Positive chunks and a single-class model are required")
    if not model.sem_seg_postprocess_before_inference:
        raise ValueError("Expected original-size logits before instance inference")
    image = record["image"].to(model.device)
    images = ImageList.from_tensors(
        [(image - model.pixel_mean) / model.pixel_std], model.size_divisibility)
    output = model.sem_seg_head(model.backbone(images.tensor))
    coarse = output["pred_masks"][0]
    classes = output["pred_logits"][0].softmax(-1)[:, 0]
    padded_size, image_size = images.tensor.shape[-2:], images.image_sizes[0]
    output_size = (record["height"], record["width"])
    masks = torch.empty((len(classes), *output_size), dtype=torch.bool, device=coarse.device)
    quality = torch.empty_like(classes)
    for start in range(0, len(classes), chunk_size):
        end = min(start + chunk_size, len(classes))
        logits = restore_logits(coarse[start:end], padded_size, image_size, output_size)
        masks[start:end], quality[start:end] = masks_and_quality(logits)
    ids = selection_indices(classes, quality, model.test_topk_per_image, "mask-first")
    scores = classes * quality
    coarse_instances = make_instances(masks, scores, ids)
    head = getattr(model.criterion, "hires_head", None)
    diagnostics = {"image_id": int(record["image_id"]), "selected_queries": len(ids),
                   "refined_queries": 0, "changed_pixels": 0,
                   "head_residual_norm": float(head.residual[-1].weight.norm()) if head is not None else None}
    if head is None:
        return {"instances": coarse_instances, "diagnostics": diagnostics}

    selected_masks = coarse_instances.pred_masks.clone()
    selected_scores = coarse_instances.scores.clone()
    areas = selected_masks.flatten(1).sum(1)
    positions = ((areas > 0) & (areas <= 1024)).nonzero(as_tuple=True)[0]
    diagnostics["refined_queries"] = len(positions)
    if len(positions):
        features = head.image_features(images.tensor)
        for start in range(0, len(positions), chunk_size):
            loc = positions[start:start + chunk_size]
            query_ids = ids[loc]
            batch_ids = torch.zeros(len(loc), dtype=torch.long, device=coarse.device)
            refined = head.refine(coarse[query_ids], features, batch_ids)
            logits = restore_logits(refined, padded_size, image_size, output_size)
            new_masks, new_quality = masks_and_quality(logits)
            diagnostics["changed_pixels"] += int((new_masks != selected_masks[loc]).sum())
            selected_masks[loc] = new_masks
            selected_scores[loc] = classes[query_ids] * new_quality
    refined_instances = make_instances(selected_masks, selected_scores,
                                      torch.arange(len(ids), device=coarse.device))
    return {"instances": refined_instances, "coarse_instances": coarse_instances,
            "diagnostics": diagnostics}


class PairedInference(nn.Module):
    """Preserve inference_on_dataset's eval/train restoration on a DDP model."""
    def __init__(self, model, chunk_size=4):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.training = model.training

    def forward(self, inputs):
        model = self.model.module if hasattr(self.model, "module") else self.model
        return [paired_prediction(model, record, self.chunk_size) for record in inputs]
