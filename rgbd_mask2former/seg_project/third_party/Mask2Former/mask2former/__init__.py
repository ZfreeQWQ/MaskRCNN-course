# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
# PATCHED (2026-10-20, CoreX compat, documented in seg_project/docs/plan.md §9): this repo's
# detectron2 build never compiles the `detectron2._C` CUDA extension (Mask R-CNN experiments use
# torchvision ops + the official pycocotools evaluator instead; see seg_project/OPERATIONS.md).
# InstanceSegEvaluator only exists to wrap detectron2's _C-accelerated COCOeval_opt, which this
# project does not use (tools/evaluate.py runs pycocotools COCOeval directly). Import lazily so
# `import mask2former` still works; fails loudly only if someone actually tries to use it.
try:
    from .evaluation.instance_evaluation import InstanceSegEvaluator
except ImportError as _e:
    def InstanceSegEvaluator(*args, **kwargs):
        raise ImportError(
            "InstanceSegEvaluator requires detectron2._C (not compiled in this environment); "
            "use seg_project/tools/evaluate.py (pycocotools COCOeval) instead"
        ) from _e
