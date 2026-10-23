# Interpretation log — 2026-10-23

This log records single-seed observations, not statistical significance.
All completed comparisons use the fixed 30k checkpoint and all 2,584 val
images, without changing validation area ranges or adding val to training.
Training source remains the immutable commit `28b288d`.

## Completed: e05_foreground_points is a negative result

| Metric | Baseline | e05 | Difference |
| --- | --- | --- | --- |
| Small-mask APS | 0.2916244869 | 0.1927241338 | -0.0989003531 |
| Overall mask AP | 0.6343571918 | 0.6044661180 | -0.0298910737 |

This variant replaces 10% of the matched small-instance mask-loss sample
points with GT foreground pixel centers. It does not change matching point
sampling, inference, or the model parameter count. Its early decline was
not used to stop the run; the planned full 30k budget was completed.

The result rejects this particular sampling recipe on this dataset/seed.
It does not establish that all foreground/boundary sampling is ineffective.
A plausible but unproven explanation is a changed foreground/background
loss-sampling distribution. Causal diagnosis would require mask-size,
false-positive/recall and boundary error analysis, not just the aggregate AP.
Do not silently reduce the sampling ratio or merge it with other changes
inside this fixed suite. Any follow-up needs its own config/run/version.

## Current trade-offs

- e01 resampling and e02 targeted cropping give small positive changes;
  repeat seeds are needed before claiming robust gains.
- e03 conservative depth-consistent Copy-Paste improves overall AP slightly,
  but its small-target APS gain is only 0.0009083444.
- e04 small-mask loss x2 currently has the best completed APS (0.2971929528),
  at a cost in overall AP (0.6291635621). It is a candidate, not a definitive
  overall winner, and no combination has been tested.
- e06 depth-guided stride-4 gating is running; e07 resolution1280 is queued.
  Do not use intermediate or smoke scores as their final comparison results.

Timing differences among training-only variants are not evidence of faster
inference. All final exports and source/initial-weight audits are recorded
by `collect_ablation_results.py` in the progress report.
