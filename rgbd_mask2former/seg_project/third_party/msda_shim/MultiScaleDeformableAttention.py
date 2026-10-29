"""Compatibility shim for Mask2Former's compiled `MultiScaleDeformableAttention` CUDA op.

Mask2Former's `ms_deform_attn_func.py` does `import MultiScaleDeformableAttention as MSDA`
and expects a compiled extension exposing `ms_deform_attn_forward` / `ms_deform_attn_backward`.
We never build that extension (no internet, and CoreX needs its own compile flags). Instead this
module forwards to the CUDA kernel already compiled inside `mmcv-2.1.0+corex.4.4.0`
(`mmcv.ops.multi_scale_deform_attn.ext_module`), which exposes the same two C++ functions under
the same names (Deformable-DETR heritage). Verified against the pure-PyTorch reference
(`multi_scale_deformable_attn_pytorch`) on this GPU: forward max abs diff 5e-7, grad relative
error ~1e-6 for value/attention-weight, ~7% max for sampling-location grad (a known boundary-
sample discrepancy between CUDA kernel and `grid_sample` backward at exact grid points, not a
correctness bug). See seg_project/docs/plan.md §2.2 / §9.
"""
import torch

from mmcv.ops.multi_scale_deform_attn import ext_module


def ms_deform_attn_forward(value, value_spatial_shapes, value_level_start_index,
                            sampling_locations, attention_weights, im2col_step):
    return ext_module.ms_deform_attn_forward(
        value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, im2col_step=im2col_step)


def ms_deform_attn_backward(value, value_spatial_shapes, value_level_start_index,
                             sampling_locations, attention_weights, grad_output, im2col_step):
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)
    ext_module.ms_deform_attn_backward(
        value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, grad_output.contiguous(),
        grad_value, grad_sampling_loc, grad_attn_weight, im2col_step=im2col_step)
    return grad_value, grad_sampling_loc, grad_attn_weight
