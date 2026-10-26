# Dataset and existing P2 error audit (no new model evaluation)

All counts recomputed from supplied val annotations or existing fixed-screen
diagnostics. They guide supervised task development, never train on val/test.
No test scores/labels, new full-val evaluation or new confirmation used.

Full val:2584 images,73421 noncrowd instances,10999 small(area<=1024),14.98%.
Small:865 area<=64(7.864%),2185 area<=256,8814 area257..1024(80.13%).
4023 small masks fill<25% of bbox(36.576%);2148 aspect ratio>3(19.529%).
Instances/image min4,median21,max96. Low fill indicates thin/irregular/occluded
shapes; it does NOT establish disconnected fragments or annotation noise.

## Fixed200 screen, trained P2 j01 final3000

Existing diagnose_small reports independent best IoU over all predictions per
GT. Multiple GTs may reuse one prediction, so this is an ORACLE diagnostic,
not AP, true detector recall, independent matching or exact causal error labels.

| GT group | total | bestIoU<.1 | .1<=IoU<.5 | .5<=IoU<.75 | >=.75 |
| --- | --- | --- | --- | --- | --- |
|all small|921|78|114|208|521|
|area<=64|68|46|8|13|1|
|area257..1024|746|13|81|165|487|
|fill<.25|314|31|75|112|96|

Counts reconstructed from stored exact fractions*group size, rounded to
integer; no per-image predictions were exported by earlier evaluator. IoU<.1
means poor overlap, NOT proven absence of any predicted object. .5..75 is a
plausible precision-refinement cohort, not proof every error is a boundary.
P2 smallAP50 .745886,AP75 .506297,AR .519327. The gap also mixes ranking and
missed-instance effects. Per-density AP/duplicate/merge diagnosis is NOT yet
available; do not invent it from annotation density alone.

## Priority and limits

Most small instances are257..1024 pixels, so boundary/detail precision there
may cover more errors than exclusively optimizing the7.9% tiniest GT. AP is
not a simple instance-count-weighted sum. A final-mask refiner cannot recover
every missed instance; if it fails, investigate learned image-only proposals/
query localization and per-density errors rather than endlessly adjust sampling.
Geometry fusion already helped at full val. Additional geometry capacity did
not automatically help; use an RGB-D vs explicit-geometry point-head ablation,
sharing common new-module initialization and original depth-dropout draw.
