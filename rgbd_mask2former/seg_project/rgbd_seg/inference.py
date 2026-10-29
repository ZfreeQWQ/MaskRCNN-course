import torch
from detectron2.modeling import build_model
from .config import setup_cfg
from .data import RGBDMapper
from .checkpoint import RGBDCheckpointer


class Predictor:
    def __init__(self, cfg, weights):
        if cfg.MODEL.DEVICE == "cpu":
            torch.set_num_threads(cfg.PROJECT.CPU_THREADS)
        self.cfg = cfg
        self.mapper = RGBDMapper(cfg, is_train=False)
        self.model = build_model(cfg)
        self.model.eval()
        RGBDCheckpointer(self.model, depth_init=cfg.PROJECT.DEPTH_INIT).load(weights)

    @torch.inference_mode()
    def __call__(self, record):
        return self.model([self.mapper(record)])[0]


def load_predictor(config_file, weights, device=None, opts=()):
    options = list(opts)
    if device:
        options += ["MODEL.DEVICE", device]
    return Predictor(setup_cfg(config_file, options), weights)
