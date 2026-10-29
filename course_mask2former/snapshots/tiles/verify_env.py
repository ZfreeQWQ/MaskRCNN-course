from pathlib import Path

import cv2
import mmcv
import mmengine
import mmdet
import torch
from mmcv.ops import nms
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"mmcv={mmcv.__version__}")
    print(f"mmengine={mmengine.__version__}")
    print(f"mmdet={mmdet.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    x = torch.randn((2048, 2048), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"cuda_matmul={tuple(y.shape)},finite={bool(torch.isfinite(y).all())}")

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8], device="cuda")
    _, keep = nms(boxes, scores, 0.5)
    print(f"mmcv_cuda_nms={keep.tolist()}")

    config_path = Path("/workspace/mmdetection/configs/mask2former")
    configs = sorted(config_path.glob("mask2former_r50*_coco.py"))
    if not configs:
        raise FileNotFoundError("Mask2Former R50 instance config was not found")
    cfg = Config.fromfile(configs[0])
    print(f"mask2former_config={configs[0].name}")
    print(f"model_type={cfg.model.type}")
    register_all_modules(init_default_scope=True)
    model = MODELS.build(cfg.model)
    print(f"model_build={model.__class__.__name__}")


if __name__ == "__main__":
    main()
