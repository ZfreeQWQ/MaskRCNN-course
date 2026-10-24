# Optimization V2 — authorized restart, 2026-10-24

User authorized continued optimization after the screenshot analysis. V1 stays
stopped after e06; its source, weights, metrics and stop controls are unchanged.
The frozen baseline is APS 0.2916244869 / AP 0.6343571918. Single-seed results
must not be called statistically significant or state of the art.

## Bounded first stage: two independent comparisons

Both start from the same permitted COCO initialization, seed42, batch2, FP32,
30k optimizer updates, LR1e-4/backbone multiplier0.1, drops18k/24k, 1024 input,
scale0.5–1.5, 100 queries, max100 output. Only train is used for backprop;
full val2584 at final30k is the reported result. Test is not accessed.

- **d01_geometry_multiscale**: retain the original RGB-D stem and add an
  independent 7-channel geometry encoder, widths32/64/128/256, GroupNorm.
  Per-scale joint channel/spatial gates modulate additive residuals at res2–5
  before the pixel decoder. Projections start at zero; first forward must equal
  baseline features exactly. Whole residual branch is dropped per image with
  p0.2 (without inverted-dropout scaling). New modules use baseLR1e-4, not the
  pre-trained backbone's1e-5. This is inspired by gated multimodal fusion,
  **not an exact reproduction of SA-Gate or CMX**.
- **d02_roi_supervision**: original architecture/inference. For originally
  small GT only, Hungarian BCE/Dice is 0.7 global +0.3 per-GT ROI; class cost is
  unchanged. ROI has256 uniform points in its own transformed bounding box,
  expanded20% in total, clipped to unpadded image then normalized to padded
  batch canvas. Loss replaces30% of its12544 points across uncertainty/random
  slots with ROI-uniform points, retaining foreground AND local background.
  Area reweighting is supported but OFF, to avoid adding another factor.

Do not combine the two, change queries/resolution/augmentation, start old e07,
add high-resolution mask features, or run TTA automatically. This queue ends
after these two final evaluations and inference-only weight exports.

## Geometry and data integrity

Seven channels: metric camera-axis Z, signed distance above estimated dominant
support plane, camera XYZ surface normal, relative depth jump, valid-depth flag.
Only image depth and supplied image.camera.K are used; no camera extrinsics,
object poses, GT centroids, model IDs or composition metadata are model inputs.
RANSAC is deterministic per image, fits a candidate support plane with a
reliability threshold, falls back to zero height when unavailable. A dominant
plane is not guaranteed to be the physical table; diagnostic fit rates are not
semantic accuracy claims. Normals crossing >5mm center-to-neighbor discontinuities
are invalidated, including isolated foreground/background pixels.

Geometry is computed BEFORE augmentation using original K. The maps are
spatially transformed synchronously with RGB-D/masks using nearest sampling;
vectors retain original camera-frame components. New geometric channels are
train-standardized before transforms, so invalid/padded pixels are zero.
Original raw depth is used for geometry (valid0<Z<2m); the original RGB-D stem
keeps its baseline1m clipping. No learned data preprocessing is fit on val.

Intrinsics were checked for ALL supplied train/val images: one unique K.
Normalization is fit to a fixed256-image evenly spaced TRAIN-only subset, with
float64 mean/std accumulation (float32 draft retained but never used formally).
On that subset, candidate plane fit passed256/256, mean sampled inlier support
0.6261; this does not prove all candidates are true support tables.
`geometry_v2_stats.json` and `geometry_v2_intrinsics.json` hashes are recorded
in each run's protocol. They must accompany inference deployment. The current
v2 mapper deliberately requires known supplied camera metadata and fails closed
on unknown IDs; test export adaptation/metadata validation is a separate step.

## Verification and reporting

Before formal training: old regression suite, synthetic height/normal/invalid
tests, exact initial feature/classifier equality, exact COCO tensor transfer,
ROI-per-GT cost tests, empty target/gradient tests, real mapper alignment, and
10-update checkpoint/reload/two-image validation for BOTH variants. Smoke scores
are only execution checks. Initial zero-projection blocks upstream branch
gradients on step1; nonzero finite encoder/gate gradients are checked by step10.

GPU0 training / GPU1 asynchronous validation, source committed and archived
under `runs/optimization_v2/source` before queue starts. Stop on any failure.
Report final APS/AP, parameters, strict-load inference weight bytes, model-only
time and **input-preprocessing+model time**, so CPU RANSAC is not hidden.
Input+model excludes RLE/COCO aggregation and is not a full submission latency.

Sources: [SA-Gate official implementation](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch),
[CMX official implementation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation),
[Mask2Former official matcher](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py).
