# Optimization V4 — 2026-10-25

User requested continued optimization. V3 completed and stopped before V4.
Best remains V2 geometry model at inference1280: full2584-val APS0.5079599315,
AP0.8065509242. V3 detail at1024 completed30k: APS0.4260616057/AP0.7714953185,
vs plain V2 at1024 APS0.4241536588/AP0.7618873310. The small APS difference
0.001907947 is single-seed, not evidence of robust improvement. Do not merge
the results across different inference scales or claim an established win.

## Equivalent input optimization

No changes to depth geometry, normalization, model computation, weights, RGB
bilinear interpolation or mask output dimensions. Cache only the installed
nearest interpolation's exact separable indices once per input shape. Gather
depth/geometry through those indices and assemble CHW directly, avoiding
repeated float tensor interpolation/transposes and the11-channel HWC temporary.
There is NO per-image full-resolution geometry cache hiding inference costs.
Cached index-probe first-call cost is recorded; normal warmup remains explicit.

Every value checked at1024/1280/1536 on three real val images each. Synthetic
seven-channel arrays verify nearest index equivalence at all pixels. A further
26-image real1280 comparison and GPU score/mask parity on two images for BOTH
trained networks passed. Paired input benchmark alternates AB/BA, excludes first
two warmups and uses the same24 images on the same CPU/thread settings:

- Old V3 input mean0.460996s, median0.461822s.
- New V4 input mean0.371747s, median0.371916s.
- Input-only mean reduction19.36%; transform/assembly0.180543->0.094909s.
- Geometry remained~0.231s. This is NOT a19.36% whole-pipeline or network speedup.
- Paired tests use warm file caches; online geometry/normalization is recomputed.

The committed snapshot reruns all regression tests, benchmark and GPU parity.
Its benchmark numbers may vary with CPU load; report its observed values too.

## Exactly two further inference comparisons; no training

| Name | Frozen checkpoint | Input | Question |
| --- | --- | --- | --- |
| h01_scale1536 | V2 d01 final30k |1536|Does additional scale improve beyond best1280?|
| h02_detail1280 | V3 g02 final30k |1280|Does final detail refinement help at the SAME1280 scale?|

Both evaluate full2584 val with compressedRLE1024 masks, at most100/image,
standard COCO small area and IoU ranges, same data split. Strict state loading
and checkpoint SHA256 audited before launch. CPU regression, equivalent inputs,
GPU parity and two-image inference smoke precede full evaluation. Smoke scores
are not model results. Jobs use separate physical GPU0/1; file locks prevent
same-GPU evaluator overlap. CPU resources remain shared, so do not claim precise
cross-run latency gains from historical timings alone; use the paired benchmark.

Immutable committed source archive under runs/optimization_v4/source. Stop on
failure; auto-stop after these two full evaluations. Do NOT start training,
TTA,1536 detail, tiling, extra data or test inference automatically. Val tuning
is not an independent hidden-test performance claim. All original best weights,
baselines and negative ablations are retained.
