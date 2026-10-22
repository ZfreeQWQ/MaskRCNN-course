"""Opt-in experiment configuration; all defaults leave the baseline unchanged."""
from detectron2.config import CfgNode as CN


def add_experiment_config(cfg):
    cfg.COURSE = CN()
    cfg.COURSE.NAME = "baseline"
    cfg.COURSE.RESAMPLE_SMALL = False
    cfg.COURSE.SMALL_AREA = 1024.0
    cfg.COURSE.CROP_PROB = 0.0
    cfg.COURSE.CROP_SIZE = 768
    cfg.COURSE.COPY_PASTE_PROB = 0.0
    cfg.COURSE.COPY_PASTE_MAX = 5
    cfg.COURSE.SMALL_LOSS_WEIGHT = 1.0
    cfg.COURSE.FOREGROUND_POINT_FRACTION = 0.0
    cfg.COURSE.DEPTH_GATE = False
    cfg.COURSE.TEST_SIZE = 1024
    # Import registers opt-in classes, without modifying upstream implementations.
    from course import modeling  # noqa: F401


def mapper(cfg, training, cache="data_cache"):
    from course.data import RGBDMapper
    if (cfg.COURSE.CROP_PROB or cfg.COURSE.COPY_PASTE_PROB or
            cfg.COURSE.SMALL_LOSS_WEIGHT != 1 or cfg.COURSE.FOREGROUND_POINT_FRACTION or
            (not training and cfg.COURSE.TEST_SIZE != 1024)):
        from course.augmentation import ExperimentMapper
        return ExperimentMapper(cfg, training, cache)
    return RGBDMapper(cfg, training)
