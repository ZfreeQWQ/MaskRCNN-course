# Optimization V3 — 2026-10-25

User authorized further optimization and continuation. Preserve best V2 d01
APS0.4241536588/AP0.7618873310, source f596c61, final30k/full2584-val.
V2 d02 ROI supervision is a negative result: APS0.2319784360/AP0.6230754423.
No new ROI/foreground sampling, loss reweighting, backbone replacement, external
data, val training, test access or automatic extra experiment is authorized by
this bounded implementation plan.

## Completed diagnostic stage (commit31d77d4)

Same best d01 weights SHA256
f33d2d37c422d6624d0e77bcb7843d4c7ca5af190241ecf1f02a24458ba8a11b,
same fixed400 evenly spaced val images, no training:

| Input | APS | Small AP50 | Small AP75 | Overall AP | Model ms | Input+model ms |
| --- | --- | --- | --- | --- | --- | --- |
|1024|0.436039|0.696356|0.468097|0.763190|71.95|409.01|
|1280|0.523106|0.759225|0.567211|0.808516|103.09|578.57|

These are VAL400 diagnostic results, NOT new full-val results. Scores cannot
be directly compared with V2 full2584 APS0.424154. Selective full evaluation
is normal validation tuning, not an independent untouched test claim.

1697 small GTs in val400: independent best mask IoU>=.50 fraction0.7773,
>=.75 fraction0.5421 at1024 (gap0.2351). Sparse masks filling<25% bbox have
fractions0.6514/0.2792. Area<=64 GTs are mostly unresolved: only0.1563
have bestIoU>=.50. This motivates detail refinement but also shows that
boundary-only refinement cannot solve all missed tiny instances. Independent
best IoU is an oracle diagnostic, NOT COCO recall, and does not prove every
partial overlap is specifically a boundary error.

Single-read input is bitwise identical to old inputs on six train/val images,
with synchronized flips and exact masks. Actual1280 tensor geometry and original
1024 output metadata tested. Removing duplicate depth decoding is an I/O change,
not an accuracy change. Stage profiling at1024: read0.048s, geometry0.231s,
transform/tensor0.058s. Do not infer a precise speedup by comparing different
subsets or hardware loads. No full-resolution geometry cache hides online cost.

## Bounded formal stage: exactly two jobs

1. **g01 full-val inference1280**, unchanged V2 d01 final weights. Report full
   APS/AP, timing, memory and max100 output1024 masks. This is an inference-scale
   change, not a newly trained architecture.
2. **g02 geometry detail**, fresh permitted COCO initialization, seed42, FP32,
   batch2, train/infer1024,100queries,30k updates, original LR1e-4 and drops18k/24k,
   same synchronized scale0.5–1.5 augmentation. Retain V2 depth fusion, add a small
   final-only stride2 detail residual from real pre-maxpool RGB stem activations
   and normalized depth geometry. Low-rank32-channel separable convolutions,
   query projection and zero output projection. No extra detail mask at any
   intermediate Transformer layer; original coarse masks still drive all masked
   attention and Hungarian matching.

The final training mask loss samples original coarse logits + detail residual
at the SAME baseline uncertainty/random points, with the SAME BCE/Dice weights
and point budget. All auxiliary losses remain unchanged; no doubled final loss,
GT-box sampling or added boundary reweight. Inference upsamples coarse and detail
separately to input resolution before adding: this preserves exact initial
scores/masks (double bilinear interpolation would not).

All original weight keys preserved. CPU equivalence/gradient tests, exact GPU
parity on real best-model outputs,10-step smoke checkpoint/reload/val2 precede
any formal job. Smoke scores are never results. Models train from COCO, not by
silently adding updates to the best30k checkpoint. New geometry/refiner use
baseLR, pretrained backbone uses0.1 multiplier.

GPU0 trains, GPU1 evaluates. A shared physical-GPU file lock serializes all
evaluators before GPU allocation so full1280 evaluation and periodic training
validation never overlap on GPU1. They may overlap with training on GPU0.
Immutable committed source snapshots; stop on failure; finish/export final
weights then STOP. Do not automatically test g02 at1280 or start1536/TTA/queries200.
