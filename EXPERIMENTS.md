# Versioned, sequential small-instance ablations

Frozen baseline: commit `463528b`, tag `baseline-rgbd-r50-30k-v1`.
Full val: APS 0.2916244869; mask AP 0.6343571918.
No anchor experiments: Mask2Former has no anchors. Stride-4 mask features
already exist; the depth gate experiment enhances them, not adds missing P2.

## Fixed comparison protocol

All experiments start from the identical audited COCO-adapted initialization,
not the fine-tuned baseline. Same seed 42, 30k updates, batch 2, FP32,
AdamW LR 1e-4, backbone LR x0.1, LR drops at 18k/24k. Final 30k checkpoint
on all 2,584 val images, official COCO segmentation API, maxDets=100,
1024x1024 original output and original GT area categories.
No val/test examples or masks in training, sampling or augmentation.
No additional pretrained weights. No automatic combinations or best-val selection.

| Order | Independent change | Explicit initial setting |
| --- | --- | --- |
| e01 | Small-instance image resampling | repeat factor 1+min(num_small/4,1), max 2 |
| e02 | Small-instance-centered crop | probability .5, 768 crop before existing scale/1024 crop |
| e03 | Depth-consistent RGB-D Copy-Paste | probability .5, max 5 native-size train donor instances |
| e04 | Small-instance BCE/Dice loss weights | x2 with normalized instance weights, all decoder losses |
| e05 | Small-instance foreground loss points | replace 10% points by GT foreground pixel centers |
| e06 | Depth/edge-guided stride-4 feature gate | 2->16->256 channels, identity init, bounded residual |
| e07 | Larger train AND inference resolution | 1280, restore original 1024 masks before evaluation |

Small flags are from original annotation mask area <=1024 pixels, not
post-crop fragments or upscaled areas. Instance loss weighting and foreground
point sampling are separate experiments; e05 does NOT change Hungarian
matching costs or matching point sampling. No inference GT leakage.

Copy-Paste copies RGB/depth/masks jointly, nearest geometry for depth/masks,
requires valid nearby receiver depth (10cm), follows front-most depth,
samples donors only from small-containing train images and receiver positions
from depth-compatible pixels,
rejects excessive host fragmentation, updates visible masks and boxes,
and caps instances at 100. It is a conservative augmentation approximation,
not a physically exact 3D renderer. Accepted paste count must be audited.

## Versioning and execution

Code stays on `wzh-dev`; small custom modules live in `course/`.
Official `mask2former/` sources remain unchanged. Base config has all new
features disabled. Every run records full resolved config, source hashes,
commit, initial-weight digest, gradient checks and dataset counts.

The queue refuses dirty source. It archives committed files into an immutable
`runs/ablations_v1/source/` snapshot; subsequent working-tree edits cannot
alter running experiments. Only read-only shared data/initial weights are linked.
Large checkpoints/cache/logs remain ignored by Git; lightweight frozen metrics
and hash manifests are tracked. No remote push is performed automatically.

Run preparation: `python prepare_experiments.py` (train-only statistics).
Queue: `python -u run_ablation_suite.py`.
Queue first runs CPU tests and 10-update/2-image GPU smoke tests for every
variant, then formal experiments in order. Smoke scores never enter the
comparison. Stop on a failure/OOM; do not silently change batch/precision/LR.
Existing incomplete runs are never overwritten or silently resumed.

GPU0 trains; GPU1 evaluates asynchronously. Intermediate val is monitoring
only. All seven full runs may take roughly 1–2 days; augmentation and 1280
costs must be measured rather than assumed equal to baseline.

Progress: `runs/ablations_v1/suite.json`, per-run `status.json`, and
`runs/ablations_v1/RESULTS.md`. Record negative results too. Small gains need
additional seeds before claiming statistically stable improvement.
