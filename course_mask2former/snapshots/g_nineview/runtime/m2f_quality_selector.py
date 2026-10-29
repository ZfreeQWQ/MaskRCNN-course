"""Train-only, fixed ridge quality gate; no torch, model, or GT at inference."""
import copy
import hashlib
import math
from pathlib import Path

SEED = "20261020-quality-v1"
FEATURES = ("global_score", "local_score", "log_global_area", "log_local_area",
            "log_area_ratio", "pair_iou", "intersection_global", "intersection_local",
            "global_fill", "local_fill", "global_log_aspect", "local_log_aspect",
            "normalized_center_distance")


def digest(value):
    return hashlib.sha256((SEED + ":" + str(value)).encode()).hexdigest()


def select_manifest(images):
    assert len({i["id"] for i in images}) == len(images)
    pools = {"fit": [], "check": []}
    for image in images:
        scene = str(image["scene_id"])
        assert scene and scene != "None"
        arm = "check" if int(digest("scene:" + scene), 16) % 4 == 0 else "fit"
        pools[arm].append(image)
    result = {}
    for arm, count in (("fit", 384), ("check", 128)):
        ordered = sorted(pools[arm], key=lambda i: (digest("image:" + str(i["id"])), i["id"]))
        assert len(ordered) >= count, "Insufficient scene-isolated official train images"
        result[arm] = ordered[:count]
    assert not ({str(i["scene_id"]) for i in result["fit"]} &
                {str(i["scene_id"]) for i in result["check"]})
    return result


def train_request(root, image):
    root = Path(root).resolve()
    rgb = (root / image["file_name"]).resolve()
    depth = (root / image["depth_file_name"]).resolve()
    assert "test" not in root.parts
    assert rgb.is_relative_to(root / "rgb/train") and depth.is_relative_to(root / "depth/train")
    assert (image["height"], image["width"]) == (1024, 1024)
    return dict(img_id=image["id"], img_path=str(rgb), depth_file=str(depth),
                depth_scale_m=image["depth_scale_m"], depth_invalid_code=image["depth_invalid_code"],
                scene=image["scene_id"])


def features(global_row, local_row):
    import numpy as np
    from pycocotools import mask as mu
    a, b = global_row["segmentation"], local_row["segmentation"]
    aa, ab = float(mu.area(a)), float(mu.area(b))
    assert 0 < aa <= 1024 and 0 < ab <= 1024
    ba, bb = np.asarray(mu.toBbox(a), dtype=float), np.asarray(mu.toBbox(b), dtype=float)
    overlap = float(mu.iou([a], [b], [0])[0, 0])
    intersection = overlap * (aa + ab) / (1 + overlap)
    distance = float(np.linalg.norm(ba[:2] + ba[2:] / 2 - bb[:2] - bb[2:] / 2))
    value = [global_row["score"], local_row["score"], math.log1p(aa), math.log1p(ab),
             math.log(ab / aa), overlap, intersection / aa, intersection / ab,
             aa / (ba[2] * ba[3]), ab / (bb[2] * bb[3]),
             math.log(ba[2] / ba[3]), math.log(bb[2] / bb[3]),
             distance / float(np.linalg.norm(ba[2:]))]
    assert len(value) == len(FEATURES) == 13 and np.isfinite(value).all()
    return value


def design(x, mean, scale):
    import numpy as np
    x = np.asarray(x, dtype=np.float64).reshape(-1, 13)
    z = (x - np.asarray(mean)) / np.asarray(scale)
    terms = [z] + [(z[:, i] * z[:, j])[:, None] for i in range(13) for j in range(i, 13)]
    phi = np.concatenate(terms, axis=1)
    assert phi.shape[1] == 104 and np.isfinite(phi).all()
    return phi


def fit(x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.ndim == 2 and x.shape == (len(y), 13) and len(y) >= 2
    assert np.isfinite(x).all() and np.isfinite(y).all() and (abs(y) <= 1).all()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    phi = design(x, mean, scale)
    pm, ym = phi.mean(axis=0), float(y.mean())
    centered = phi - pm
    # mean squared error + lambda*||w||², lambda exactly 1; intercept unpenalized.
    coef = np.linalg.solve(centered.T @ centered + len(y) * np.eye(104),
                           centered.T @ (y - ym))
    intercept = ym - float(pm @ coef)
    assert np.isfinite(coef).all() and math.isfinite(intercept)
    return dict(kind="fixed_quadratic_ridge", lambda_l2=1., threshold=0.,
                feature_names=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(),
                coefficients=coef.tolist(), intercept=intercept, fitted_pairs=len(y),
                trainable_parameters=105, numpy_dtype="float64", seed=SEED)


def predict(x, selector):
    import numpy as np
    assert selector["feature_names"] == list(FEATURES)
    assert selector["lambda_l2"] == 1. and selector["threshold"] == 0.
    result = design(x, selector["mean"], selector["scale"]) @ np.asarray(selector["coefficients"])
    result += selector["intercept"]
    assert np.isfinite(result).all()
    return result


def choose(base, merged, selector):
    """Use the frozen original candidate choice, gate only its geometry."""
    import m2f_cached_geometry_refine as original
    refined, changes = original.refine(base, merged)
    if not changes:
        return refined, []
    values = predict([features(base[c["index"]], merged[c["local_index"]]) for c in changes], selector)
    decisions = []
    for change, value in zip(changes, values):
        accepted = bool(value > 0.)
        if not accepted:
            refined[change["index"]] = copy.deepcopy(base[change["index"]])
        decisions.append(dict(**change, predicted_delta=float(value), accepted=accepted))
    assert len(refined) == len(base)
    for old, new in zip(base, refined):
        assert {k: v for k, v in old.items() if k not in ("bbox", "segmentation")} == {
            k: v for k, v in new.items() if k not in ("bbox", "segmentation")}
    return refined, decisions


def supervised_pairs(base, merged, annotations, coco):
    """A fixed same-GT delta, never two independently chosen GT identities."""
    import numpy as np
    from pycocotools import mask as mu
    import m2f_cached_geometry_refine as original
    _, changes = original.refine(base, merged)
    result = []
    for c in changes:
        g, local = base[c["index"]], merged[c["local_index"]]
        valid = sorted((a for a in annotations if a["category_id"] == g["category_id"]
                        and not a.get("iscrowd", 0) and not a.get("ignore", 0)), key=lambda a: a["id"])
        rows = [coco.annToRLE(a) for a in valid]
        gt_id, area, before, after = None, None, 0., 0.
        if rows:
            values = mu.iou([g["segmentation"], local["segmentation"]], rows, [0] * len(rows))
            j = int(np.argmax(values[0]))
            if values[0, j] > 0:
                gt_id, area = valid[j]["id"], valid[j]["area"]
                before, after = float(values[0, j]), float(values[1, j])
        result.append(dict(**c, image_id=g["image_id"], features=features(g, local),
                           gt_id=gt_id, gt_area=area, global_iou=before, local_iou=after,
                           target_delta=after-before))
    return result


def diagnostic(pairs, selector):
    import numpy as np
    if not pairs:
        return dict(pairs=0)
    pred = predict([p["features"] for p in pairs], selector)
    y = np.asarray([p["target_delta"] for p in pairs])
    accepted = pred > 0
    return dict(pairs=len(pairs), mse=float(np.mean((pred-y)**2)),
                accepted=int(accepted.sum()), rejected=int((~accepted).sum()),
                harmful_accepted=int(((y < 0) & accepted).sum()),
                beneficial_rejected=int(((y > 0) & ~accepted).sum()),
                beneficial_accepted=int(((y > 0) & accepted).sum()),
                harmful_rejected=int(((y < 0) & ~accepted).sum()),
                no_gt_intersection=sum(p["gt_id"] is None for p in pairs),
                realized_iou_delta_sum=float(y[accepted].sum()),
                original_always_replace_delta_sum=float(y.sum()))
