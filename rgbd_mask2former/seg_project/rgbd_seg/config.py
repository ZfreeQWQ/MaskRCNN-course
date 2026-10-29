from pathlib import Path
from detectron2.config import CfgNode as CN, get_cfg as d2_get_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_cfg():
    cfg = d2_get_cfg()
    cfg.PROJECT = CN()
    cfg.PROJECT.EXPERIMENT = "B0"
    cfg.PROJECT.DATASET_ROOT = str(PROJECT_ROOT / "datasets/blender")
    cfg.PROJECT.DEPTH_STATS = str(PROJECT_ROOT / "datasets/blender/annotations/depth_stats.json")
    cfg.PROJECT.USE_DEPTH = False
    cfg.PROJECT.DEPTH_INIT = "mean"
    cfg.PROJECT.CPU_THREADS = 4
    cfg.PROJECT.EVAL_LIMIT = 0
    cfg.PROJECT.SKIP_FINAL_EVAL = False
    return cfg


def setup_cfg(config_file, opts=()):
    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.merge_from_list(list(opts))
    channels = 4 if cfg.PROJECT.USE_DEPTH else 3
    if len(cfg.MODEL.PIXEL_MEAN) != channels or len(cfg.MODEL.PIXEL_STD) != channels:
        raise ValueError("PIXEL_MEAN/STD channel count must agree with USE_DEPTH")
    if cfg.DATASETS.TEST != ("blender_val",):
        raise ValueError("Model selection/evaluation is restricted to blender_val")
    if cfg.DATASETS.TRAIN != ("blender_train",):
        raise ValueError("Training is restricted to blender_train")
    if cfg.PROJECT.USE_DEPTH and cfg.MODEL.BACKBONE.FREEZE_AT > 0:
        raise ValueError("RGB-D stem must remain trainable: set BACKBONE.FREEZE_AT=0")
    cfg.freeze()
    return cfg
