"""Config setup for Mask2Former training/inference on this project's RGB-D data.

Mirrors rgbd_seg/config.py's PROJECT.* convention; add_maskformer2_config comes from the
uploaded third_party/Mask2Former checkout (see docs/plan.md for the layout).
"""
from pathlib import Path

from detectron2.config import CfgNode as CN, get_cfg as d2_get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_cfg():
    cfg = d2_get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.PROJECT = CN()
    cfg.PROJECT.EXPERIMENT = "B"
    cfg.PROJECT.DATASET_ROOT = str(PROJECT_ROOT / "datasets/blender")
    cfg.PROJECT.DEPTH_GEOMETRY_STATS = str(PROJECT_ROOT / "datasets/blender/annotations/depth_geometry_stats.json")
    cfg.PROJECT.USE_DEPTH = False  # I1: geometric depth encoding + gated fusion
    cfg.PROJECT.DEPTH_WIDTHS = [32, 64, 128, 256]
    cfg.PROJECT.DEPTH_DROPOUT_P = 0.2
    cfg.PROJECT.CPU_THREADS = 4
    cfg.PROJECT.EVAL_LIMIT = 0
    cfg.PROJECT.SKIP_FINAL_EVAL = False
    # I2: small-object-aware point sampling in the matcher/criterion (no inference-time cost).
    cfg.PROJECT.SMALL_OBJECT_POINTS = CN()
    cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED = False
    cfg.PROJECT.SMALL_OBJECT_POINTS.NUM_POINTS = 256
    cfg.PROJECT.SMALL_OBJECT_POINTS.BOX_EXPAND = 0.2
    cfg.PROJECT.SMALL_OBJECT_POINTS.LOSS_POINT_FRACTION = 0.3
    cfg.PROJECT.SMALL_OBJECT_POINTS.AREA_WEIGHT_ENABLED = False
    cfg.PROJECT.SMALL_OBJECT_POINTS.AREA_WEIGHT_REF = 1024.0
    cfg.PROJECT.SMALL_OBJECT_POINTS.AREA_WEIGHT_POWER = 0.25
    cfg.PROJECT.SMALL_OBJECT_POINTS.AREA_WEIGHT_MAX = 2.0
    # I3: depth-guided 1/2-resolution mask feature branch.
    cfg.PROJECT.HIRES_MASK = CN()
    cfg.PROJECT.HIRES_MASK.ENABLED = False
    cfg.PROJECT.HIRES_MASK.USE_DEPTH = True
    # Hard cap on matched pairs refined per loss_masks() call (memory ceiling; see
    # rgbd_seg/m2f/criterion.py's HiResCriterion docstring for the OOM this prevents).
    cfg.PROJECT.HIRES_MASK.MAX_PAIRS_PER_STEP = 24
    # Our GT segmentation is compressed RLE (bitmask), not polygons.
    cfg.INPUT.MASK_FORMAT = "bitmask"
    # Augmentation knobs the official configs don't expose.
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # "horizontal" | "vertical" | "both" | "none"
    cfg.INPUT.ROTATE_90 = False
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.MIN_SCALE = 0.8
    cfg.INPUT.MAX_SCALE = 1.6
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.TEST_IMAGE_SIZE = 1024
    return cfg


def setup_cfg(config_file, opts=()):
    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.merge_from_list(list(opts))
    if cfg.DATASETS.TEST != ("blender_val",):
        raise ValueError("Model selection/evaluation is restricted to blender_val")
    if cfg.DATASETS.TRAIN != ("blender_train",):
        raise ValueError("Training is restricted to blender_train")
    if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES != 1:
        raise ValueError("This dataset has a single category (component); set NUM_CLASSES=1")
    cfg.freeze()
    return cfg
