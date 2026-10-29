"""Native-pixel RGB-D mask correction; no GT is accepted by inference APIs."""
import copy
import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.nn import functional as F

PATCH = 128
RADIUS = 4
SEED = 20261013
STEPS = 3000
BATCH = 32


def encode(mask):
    rle = mu.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def window(row):
    """Native 128px crop, shifted at image edges; never resized or GT-selected."""
    rle = row["segmentation"]
    assert rle["size"] == [1024, 1024]
    area = int(mu.area(rle))
    if not 0 < area <= 1024:
        return None
    x, y, w, h = map(int, mu.toBbox(rle))
    if w + 2 * RADIUS > PATCH or h + 2 * RADIUS > PATCH:
        return None
    x0 = min(1024 - PATCH, max(0, (2*x+w-PATCH)//2))
    y0 = min(1024 - PATCH, max(0, (2*y+h-PATCH)//2))
    assert x0 <= max(0, x-RADIUS) and min(1024, x+w+RADIUS) <= x0+PATCH
    assert y0 <= max(0, y-RADIUS) and min(1024, y+h+RADIUS) <= y0+PATCH
    return x0, y0


def band(mask):
    """Chebyshev radius4 inside/outside band, background beyond image/crop."""
    m = np.asarray(mask, np.uint8)
    kernel = np.ones((2*RADIUS+1, 2*RADIUS+1), np.uint8)
    hi = cv2.dilate(m, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    lo = cv2.erode(m, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (hi != lo)


def load_rgbd(request):
    rgb = cv2.imread(request["img_path"], cv2.IMREAD_COLOR)
    raw = cv2.imread(request["depth_file"], cv2.IMREAD_UNCHANGED)
    assert rgb is not None and raw is not None
    assert rgb.shape == (1024, 1024, 3) and raw.shape == (1024, 1024)
    assert raw.dtype == np.uint16
    scale = float(request["depth_scale_m"])
    assert np.isfinite(scale) and scale > 0
    valid = (raw != request["depth_invalid_code"]) & (raw > 0)
    depth = np.where(valid, raw.astype(np.float32)*scale, 0).astype(np.float32)
    assert np.isfinite(depth).all()
    return rgb[:, :, ::-1].copy(), depth, valid


def patch_input(rgb, depth, valid, coarse):
    """Fixed unit conversion, not statistics estimated from val."""
    x = np.concatenate((rgb.astype(np.float32)/255.,
                        np.log1p(depth)[..., None],
                        valid[..., None].astype(np.float32),
                        coarse[..., None].astype(np.float32)), axis=2)
    assert x.shape == (PATCH, PATCH, 6) and np.isfinite(x).all()
    return x.transpose(2, 0, 1).copy()


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU())


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Block(6, 16)
        self.enc2 = Block(16, 32)
        self.middle = Block(32, 64)
        self.dec2 = Block(96, 32)
        self.dec1 = Block(48, 16)
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.middle(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear",
                                               align_corners=False), b), 1))
        e = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="bilinear",
                                               align_corners=False), a), 1))
        # Exact identity at initialization, residual logits only.
        return self.out(e)[:, 0] + (x[:, 5]*2-1)*2.


def band_loss(logits, target, changeable):
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    weights = changeable.float()
    assert bool((weights.sum((1, 2)) > 0).all())
    return ((loss*weights).sum((1, 2))/weights.sum((1, 2))).mean()


def compose(row, xy, coarse, changeable, proposal):
    result = copy.deepcopy(row)
    edited = coarse.copy()
    edited[changeable] = proposal[changeable]
    fallback = not bool(edited.any())
    if fallback:
        edited = coarse
    original = mu.decode(row["segmentation"]).astype(bool)
    x, y = xy
    original[y:y+PATCH, x:x+PATCH] = edited
    assert original.any()
    if not np.array_equal(edited, coarse):
        result["segmentation"] = encode(original)
        if "bbox" in result:
            result["bbox"] = mu.toBbox(result["segmentation"]).tolist()
    return result, fallback


def infer(rows, rgbd, model, device):
    """Fixed padded32 microbatch ensures cache/live numerical path agrees."""
    rgb, depth, valid = rgbd
    output = copy.deepcopy(rows)
    jobs = []
    for i, row in enumerate(rows):
        xy = window(row)
        if xy is None:
            continue
        x, y = xy
        coarse = mu.decode(row["segmentation"])[y:y+PATCH, x:x+PATCH].astype(bool)
        changeable = band(coarse)
        jobs.append((i, xy, coarse, changeable,
                     patch_input(rgb[y:y+PATCH, x:x+PATCH],
                                 depth[y:y+PATCH, x:x+PATCH],
                                 valid[y:y+PATCH, x:x+PATCH], coarse)))
    stats = dict(eligible=len(jobs), changed=0, empty_fallback=0)
    with torch.inference_mode():
        for start in range(0, len(jobs), BATCH):
            group = jobs[start:start+BATCH]
            data = np.zeros((BATCH, 6, PATCH, PATCH), np.float32)
            for k, item in enumerate(group):
                data[k] = item[4]
            predicted = (model(torch.from_numpy(data).to(device)) > 0).cpu().numpy()
            for k, (i, xy, coarse, changeable, _) in enumerate(group):
                output[i], fallback = compose(rows[i], xy, coarse, changeable, predicted[k])
                stats["empty_fallback"] += int(fallback)
                stats["changed"] += int(output[i]["segmentation"] != rows[i]["segmentation"])
    assert len(rows) == len(output)
    for a, b in zip(rows, output):
        assert {k:v for k,v in a.items() if k not in ("segmentation", "bbox")} == {
            k:v for k,v in b.items() if k not in ("segmentation", "bbox")}
    return output, stats
