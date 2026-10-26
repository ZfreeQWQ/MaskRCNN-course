# Retained P2 frozen-screen bottleneck audit

2026-10-26. Input is the trained j01 P2 final3000 checkpoint, unchanged frozen
200-image screen (100 scenes), 1024 input, strict load. No training, independent
confirmation, full val or test. Source 01a8915, weights SHA256
ac49c8e873526b993e9cb808a156258e76629697c639fd780c43ce0e2f5e1ebf.
Raw predictions and full per-GT/per-image diagnostics are persisted in
`runs/instance_audit_v1_r2/{predictions,metrics}.json`.

## Failures and independent verification

v1 failed when COCO.loadRes mutated the input with ndarray bboxes and a second
load hit an incompatible bbox comparison. v1_r2 fixed that and completed all
inference/COCO/error calculations and wrote metrics/predictions, but its wrapper
post-check requested mask_ap from the small-only diagnostic summary and raised
KeyError. Keep both failed statuses/logs/sources unchanged. Separate offline
verification must compare produced AP/APs/AP75/AR/panel against the old final
reference; no need to repeat inference to repair a wrapper-only check.
This offline verification PASSED and is recorded in verification.json.
All four metrics reproduce within1e-10, the frozen panel hash and200 image IDs
match, and20000 persisted predictions were checked. Metrics SHA256
7ccd1c1a29f8cb4397982491e3d269696c619ac608753faf91b43d61d015650e;
prediction SHA256 b86b397eb15dc844425950df2e4e3e016a1c514706b3f8373f284022d0a3dc45.

## Reproduced primary metrics

APs .47059152731761617; overall AP .7855258200403361;
small AP50 .7458860981710358; AP75 .5062965451185554; AR .5193268186753528.

| IoU threshold | Small GT with an adequate mask | Official one-to-one matches |
| --- | --- | --- |
| .50 | 729 / 921 | 729 / 921 |
| .75 | 521 / 921 | 521 / 921 |
| .85 | 303 / 921 | not extracted |
| .90 | 143 / 921 | 143 / 921 |
| .95 | 10 / 921 | 10 / 921 |

The high-IoU coverage drop is directly measured. These are not AP values or
universal architecture bounds. Oracle coverage can normally reuse predictions;
agreement here argues against query competition/count as the primary current
loss at these thresholds, not that no localization errors exist.

Tiny<=64:14/68 reach .50,1/68 reaches .75, none reaches .85.
Area257..1024:652/746 reach .50,487/746 reach .75,141/746 reach .90.
Low-fill<.25:208/314 reach .50,96/314 reach .75,7/314 reach .90.
Only14 GT with an IoU>=.50 candidate have all adequate scores below .1.
At threshold .50, matched detection median score is .9776572.

Diagnostic max-IoU re-ranking of identical predictions gives APs .4094502915,
lower by .0611412358. It uses GT, is not deployable or a rigorous AP bound,
and duplicate predictions sharing high IoU can harm such ranking. This does
not prove scoring is optimal; it does not support prioritizing this naive probe.

## Density and heuristic errors

| Noncrowd GT/image | Images | Small GT | APs | Small AR |
| --- | --- | --- | --- | --- |
| <=20 | 92 | 63 | .4440664414 | .5412698413 |
| 21..40 | 60 | 204 | .4512835882 | .5377450980 |
| >40 | 48 | 654 | .4805695489 | .5114678899 |

Different sizes/shapes/scene composition confound density comparisons. Dense
small AR is somewhat lower but small APs is not lower, so do not claim density
alone explains the primary deficit or automatically double queries.
There are5953 GT total,17356 nonempty predictions. At score>=.1,171 extra masks
share a best GT at IoU>=.5, and79 masks cover>=50% of at least two GT while
best IoU<.5. These are all-size heuristics, not causal duplicate/merge labels.

## Next controlled training screen

Do not stack rejected point/floor/geometry modules. Retain P2+existing geometry,
original matching/point coordinates and mask representation. Test normalized
small-instance loss emphasis, because small GT are only14.98% of all val GT,
the high-IoU deficit is measured, and the old RGB-D-only e04 x2 experiment had
a modest positive full-val APs change. That old result is motivation, not proof
of additivity with stronger geometry/P2. No new loss factor is validated yet.
Short +3000 adaptation is a screen, not a final convergence ceiling.
