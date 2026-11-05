"""Portable bindings for the RGB-D MaskDINO inference implementation."""
import functools
from pathlib import Path

import settings
import geometry_maskdino as r
from PIL import Image
from native_quality_features import NAMES

t, np = r.torch, r.np
EXPECTED = {
    'crop.pth': '472463ef32044a3525417bcfe4c5ff162af424aa6e704609799bde1242ec88de',
    'edge_weights.pth': 'c06034a04101520359c02802ded3e3a9fd512737f2d70a8be467e786cb16533b',
    'parent_ordinal_weights.pth': 'd433f356a495e46c51d33f6785d437237515e2a4e2133a6a062d0f6c09136680',
    'soft_quality_weights.pth': 'dfa7ee8309a4b6a4528b3dc7730cbededd6370ec9b67fadbb04f124f8a9ce813',
}


def weight_paths():
    return {name: (settings.WEIGHTS if name == 'crop.pth' else settings.POSTPROCESS_WEIGHTS)/name
            for name in EXPECTED}


def verify_weights():
    for name, path in weight_paths().items():
        digest = r.d.sha(path)
        assert digest == EXPECTED[name], 'Incorrect release weight: '+name
    return dict(EXPECTED)


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verify_weights()
    t.set_num_threads(4)
    t.cuda.set_per_process_memory_fraction(.85)
    model, _ = r.build(out, 'geometry')
    ck = t.load(settings.WEIGHTS/'crop.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    del ck
    model.eval()
    # Keep the original model's 100 predictions in each view.
    assert model.num_queries == 300 and model.test_topk_per_image == 100
    return model


def native(data, i):
    row = data.index.records[i]
    item = data.base.data.item(i, targets=False, flip=False)
    with Image.open(data.index.path(row['depth_file_name'])) as im:
        raw = np.array(im)
    return item['image'], raw.astype(np.float32)*row['depth_scale_m'], raw != row['depth_invalid_code'], data.cameras[row['id']]


@functools.lru_cache(maxsize=1)
def load_parent_head():
    ck = t.load(settings.POSTPROCESS_WEIGHTS/'parent_ordinal_weights.pth', map_location='cpu', weights_only=False)
    assert tuple(ck['features']) == NAMES and len(ck['thresholds']) == 10
    model = t.nn.Sequential(t.nn.Linear(23, 64), t.nn.ReLU(), t.nn.Linear(64, 64),
                            t.nn.ReLU(), t.nn.Linear(64, 10), t.nn.Sigmoid()).cuda()
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model, np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)
