"""Single change: equal total fit weight for three train-GT area strata."""
import numpy as np
import m2f_quality_selector as original


def groups_and_weights(pairs):
    groups = np.asarray([0 if p["gt_area"] is not None and p["gt_area"] <= 256
                         else 1 if p["gt_area"] is not None and p["gt_area"] <= 1024
                         else 2 for p in pairs])
    counts = np.bincount(groups, minlength=3)
    assert len(pairs) and (counts > 0).all(), "All three fixed train groups required"
    weights = len(pairs) / (3. * counts[groups])
    assert np.isfinite(weights).all() and np.allclose(weights.sum(), len(pairs))
    return groups, counts, weights


def fit_weighted(x, y, weights):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    assert x.shape == (len(y), 13) and w.shape == y.shape and len(y) >= 2
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()
    assert (w > 0).all() and (abs(y) <= 1).all() and np.isclose(w.sum(), len(y))
    # Deliberately unchanged unweighted standardizer; only the loss weights vary.
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    phi = original.design(x, mean, scale)
    pm, ym = np.average(phi, axis=0, weights=w), float(np.average(y, weights=w))
    centered = phi - pm
    coef = np.linalg.solve(centered.T @ (w[:, None] * centered) + len(y)*np.eye(104),
                           centered.T @ (w * (y - ym)))
    intercept = ym - float(pm @ coef)
    assert np.isfinite(coef).all() and np.isfinite(intercept)
    return dict(kind="fixed_quadratic_ridge_group_balanced", lambda_l2=1., threshold=0.,
                feature_names=list(original.FEATURES), mean=mean.tolist(), scale=scale.tolist(),
                coefficients=coef.tolist(), intercept=intercept, fitted_pairs=len(y),
                trainable_parameters=105, numpy_dtype="float64", seed=original.SEED,
                groups=["tiny_le256", "small_257_1024", "large_or_unmatched"],
                inference_group_label_used=False)


def fit(pairs):
    _, counts, weights = groups_and_weights(pairs)
    result = fit_weighted([p["features"] for p in pairs], [p["target_delta"] for p in pairs], weights)
    result["group_counts"] = counts.tolist()
    result["group_weight_per_pair"] = (len(pairs)/(3.*counts)).tolist()
    return result
