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
    cfg.COURSE.V2 = CN()
    cfg.COURSE.V2.ENABLED = False
    cfg.COURSE.V2.GEOMETRY = False
    cfg.COURSE.V2.DEPTH_DROPOUT = 0.2
    cfg.COURSE.V2.NEW_MODULE_LR_MULTIPLIER = 1.0
    cfg.COURSE.V2.ROI_POINTS = 256
    cfg.COURSE.V2.ROI_MATCH_WEIGHT = 0.0
    cfg.COURSE.V2.ROI_LOSS_FRACTION = 0.0
    cfg.COURSE.V2.AREA_WEIGHT = False
    cfg.COURSE.V3 = CN()
    cfg.COURSE.V3.ENABLED = False
    cfg.COURSE.V3.REFINE = False
    cfg.COURSE.V3.REFINE_CHANNELS = 32
    # Import registers opt-in classes, without modifying upstream implementations.
    from course import modeling  # noqa: F401
    from course import modeling_v2  # noqa: F401
    from course import refinement_v3  # noqa: F401


def mapper(cfg, training, cache="data_cache"):
    from course.data import RGBDMapper
    if cfg.COURSE.V3.ENABLED:
        assert cfg.COURSE.V2.ENABLED and cfg.COURSE.V2.GEOMETRY
        from course.data_v3 import FastGeometryMapper
        return FastGeometryMapper(cfg, training, cache)
    if cfg.COURSE.V2.ENABLED:
        from course.data_v2 import GeometryROIMapper
        return GeometryROIMapper(cfg, training, cache)
    if (cfg.COURSE.CROP_PROB or cfg.COURSE.COPY_PASTE_PROB or
            cfg.COURSE.SMALL_LOSS_WEIGHT != 1 or cfg.COURSE.FOREGROUND_POINT_FRACTION or
            (not training and cfg.COURSE.TEST_SIZE != 1024)):
        from course.augmentation import ExperimentMapper
        return ExperimentMapper(cfg, training, cache)
    return RGBDMapper(cfg, training)
