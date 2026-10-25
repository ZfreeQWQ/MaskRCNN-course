import json
import os
from pathlib import Path


def cpu_state_dict(model):
    """Preserve PyTorch module versions used by upstream load migrations.

    A plain dict comprehension silently drops OrderedDict._metadata and makes
    MaskFormerHead treat a current checkpoint as its historical v1 format.
    """
    state = model.state_dict()
    for key, value in state.items():
        state[key] = value.detach().cpu()
    return state


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def configuration(path, overrides=()):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    from course.experiments import add_experiment_config
    add_experiment_config(cfg)
    cfg.merge_from_file(path)
    cfg.merge_from_list(list(overrides))
    cfg.freeze()
    return cfg
