# Train-only audit: tiny GT signal at uniform matcher points

Performed during screen_v4; no running model, source snapshot, optimizer or
dataset split was modified. No GPU/network forward and no val/test labels used.
This is exploratory diagnosis, not a COCO metric or an actual training replay.

## Reproducible method

-128 train records at unique linspace(0,14431,128) indices;76 contain small GT.
- Noncrowd original area0<area<=1024;341 GT total,48 area<=64,103 area<=256.
- Decode only those masks with existing decode_annotations. Generate transforms
  with the actual RGBDMapper flip/ResizeScale/FixedSizeCrop chain on a zero RGB
  canvas (these transforms depend only on shape). Apply identical segmentation
  warps used by GeometryROIMapper. No RGB or depth model inference is needed.
- Compare MIN_SCALE.5 and.8 with identical per-image numpy seed(image_id+42)
  modulo2^32;MAX_SCALE1.5,input/crop1024. This pairs image-level draws, not the
  complete multiworker training RNG trajectory.
- Per-image numpy default_rng(image_id+81) generates one32768x2 float32 uniform
  normalized point set;12544 is its prefix. Repeat the same points over all GT,
  as in the official uniform matcher. CPU PointRend point_sample bilinear,
  align_corners=False. Track max GT label>0 and>.5 separately.
- This audits the matching point set. Mask-loss uncertainty-selected points use
  another distribution and are NOT measured here. No detector recall claim.

## Tiny original area<=64

| Min scale | Retained after transform | No positive label at12544 | No positive at32768 | No label>.5 at12544 | No label>.5 at32768 |
| --- | --- | --- | --- | --- | --- |
| .5 | 44/48 | 29/44 | 16/44 | 34/44 | 26/44 |
| .8 | 45/48 | 27/45 | 17/45 | 30/45 | 22/45 |

Under the baseline scale distribution,65.9% of retained tiny GT have zero
positive label at the original uniform matching points in this audit. More
points reduce that fraction to36.4%, but many remain weakly represented.

## All original small area<=1024

| Min scale | Retained | No positive label at12544 | No positive at32768 | Mean retained pixel area |
| --- | --- | --- | --- | --- |
| .5 | 334/341 | 45/334 | 18/334 | 481.92 |
| .8 | 332/341 | 34/332 | 18/332 | 628.997 |

Area<=256 retained99/103 in both: zero-positive counts38->18 (.5) and34->18
(.8) as point budget increases. Strong-label(>.5) misses48->30 and38->23.

## Interpretation and boundary

This supports investigating sparse target visibility in the matcher, rather
than endlessly adding geometry branches. It does not establish that coverage
is the sole cause of AP errors, nor that larger point budget/min-scale will
improve final AP. Geometry resolution, decoder localization, ranking and
uncertainty-sampled mask losses remain possible bottlenecks. Finish screen_v4
before promoting or discarding either training factor; no changes mid-run.
