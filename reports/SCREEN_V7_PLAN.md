# V7: normalized small-mask supervision, then next submission

Motivation: INSTANCE_BOTTLENECK_AUDIT.md. High-IoU coverage, not primarily query
capacity, is the measured bottleneck. Old RGB-D e04 x2 gave a modest positive
APs change before geometry/P2; additive benefit is unknown and must be tested.
Do not call this new architecture or stack rejected refiners/sampling floors.

Both candidates retain j01 P2+geometry and its inference graph/parameters.
- p01: existing CourseMaskFormer / SmallInstanceCriterion, all originally-small
  targets get BCE/Dice weight2, others1; denominator num_masks*mean(all target
  weights). This includes unmatched GT in normalization, as original code.
- p02: existing ROIMaskFormer / ROICriterion, area weight
  clip((1024/original_area)^.25,1,2) with the same all-target normalizer. ROI
  matching/loss fractions remain0. Weights mildly emphasize257..1024px targets
  while capping the tiniest. No added points, new point locations or losses.

Both reweight final and auxiliary mask losses only; original class loss and
Hungarian costs unchanged. Metadata is original annotation area before scale/
crop, from train only. Inference is GT-free. Declare the meta-architecture
adapter and exactly one weight config as protocol exceptions. No other data,
LR, seed, loss coefficient, resolution, queries or sampling changes.

Same d01 30k weights and fresh P2 initialization, additional3000 updates,
fresh AdamW, shared1e-5/backbone1e-6/newP2 1e-4, seed42, batch2 FP32,
flip/scale.5..1.5/crop1024, original12544 uncertainty+random points.
Frozen200 screen at0/1000/2000/3000; matched j01 final .4705915273 is reference.
No full val, test or consumed independent panel. Require all shared tensors,
RNG, original matcher/sampler and GPU initial prediction equivalence, finite
loss/gradient tests and real10-update smokes with actual small reweighting.
Inference checkpoint export preserves OrderedDict._metadata.

Select final fixed budget, not a midpoint peak. If neither direction has
consistent gain retain j01. Finish this screen before the requested new CSV;
then follow NEXT_SUBMISSION_PLAN.md rather than indefinitely start more rounds.

Preflight status at source sync:108 CPU regression tests PASSED. This is still
an unvalidated optimization candidate: GPU initial-equivalence and real10-step
training smoke must pass before formal V7 starts. No APs benefit is claimed.
