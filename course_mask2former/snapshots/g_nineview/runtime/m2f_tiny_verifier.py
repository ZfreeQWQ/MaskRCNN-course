"""Independent tiny-entry inference. No GT accepted by these APIs."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pycocotools import mask as mu
import m2f_boundary_refiner as ref


class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in ((6, 16), (16, 32), (32, 64)):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1),
                       nn.GroupNorm(4, b), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        features = self.features(x)
        mask = F.interpolate(x[:, 5:6], size=features.shape[-2:], mode="area")
        inside = (features*mask).sum((2, 3))/mask.sum((2, 3)).clamp_min(1e-8)
        context = (features*(1-mask)).sum((2, 3))/(1-mask).sum((2, 3)).clamp_min(1e-8)
        return self.output(torch.cat((inside, context), 1))[:, 0]


def eligible(current, merged):
    jobs = []
    for index, row in enumerate(merged):
        if not 0 < int(mu.area(row["segmentation"])) <= 256:
            continue
        xy = ref.window(row)
        if xy is None:
            continue
        same = [r["segmentation"] for r in current
                if r["category_id"] == row["category_id"]]
        overlap = float(mu.iou([row["segmentation"]], same, [0]*len(same)).max()) if same else 0.
        if overlap < .5:
            jobs.append(dict(index=index, xy=list(xy), max_current_iou=overlap))
    return jobs


def patch(row, xy, rgbd):
    rgb, depth, valid = rgbd
    x, y = xy
    mask = mu.decode(row["segmentation"])[y:y+128, x:x+128].astype(bool)
    assert mask.any()
    return ref.patch_input(rgb[y:y+128, x:x+128], depth[y:y+128, x:x+128],
                           valid[y:y+128, x:x+128], mask)


def target(max_iou, gt_area):
    if gt_area is not None and gt_area <= 256 and max_iou >= .5:
        return 1
    if gt_area is None or max_iou < .1 or gt_area > 256:
        return 0
    return -1


def merge(current, merged, jobs, probabilities):
    assert len(jobs) == len(probabilities)
    pool = [dict(row=copy.deepcopy(r), origin="original", index=i)
            for i, r in enumerate(current)]
    additions = []
    for job, p in zip(jobs, probabilities):
        p = float(p)
        assert np.isfinite(p) and 0 <= p <= 1
        if p < .5:
            continue
        row = copy.deepcopy(merged[job["index"]])
        row["score"] = float(row["score"])*p
        additions.append(dict(row=row, origin="new", index=job["index"], probability=p))
    additions.sort(key=lambda r: (-r["row"]["score"], r["index"]))
    pool += additions[:5]
    pool.sort(key=lambda r: (-r["row"]["score"], r["origin"] != "original", r["index"]))
    kept, suppressed = [], []
    for item in pool:
        # Do not introduce an extra original-vs-original NMS pass.
        same = [p for p in kept if p["row"]["category_id"] == item["row"]["category_id"]
                and not (p["origin"] == item["origin"] == "original")]
        overlap = mu.iou([item["row"]["segmentation"]],
                         [p["row"]["segmentation"] for p in same], [0]*len(same)) if same else []
        if len(same) and float(np.max(overlap)) > .5:
            suppressed.append(dict(origin=item["origin"], index=item["index"], reason="mask_nms"))
        else:
            kept.append(item)
    dropped = [dict(origin=p["origin"], index=p["index"], reason="top100") for p in kept[100:]]
    kept = kept[:100]
    return [p["row"] for p in kept], dict(eligible=len(jobs), accepted=len(additions),
        proposed=min(5, len(additions)), kept=[{k:v for k,v in p.items() if k!="row"} for p in kept],
        suppressed=suppressed, top100_dropped=dropped)


def infer(current, merged, rgbd, model, device):
    jobs = eligible(current, merged)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(jobs), 32):
            group = jobs[start:start+32]
            data = np.zeros((32, 6, 128, 128), np.float32)
            for j, job in enumerate(group):
                data[j] = patch(merged[job["index"]], job["xy"], rgbd)
            out = torch.sigmoid(model(torch.from_numpy(data).to(device))).cpu().numpy()
            probabilities.extend(map(float, out[:len(group)]))
    result, detail = merge(current, merged, jobs, probabilities)
    detail["candidates"] = [dict(**job, probability=p) for job,p in zip(jobs, probabilities)]
    return result, detail

