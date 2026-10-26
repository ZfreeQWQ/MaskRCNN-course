# V7: normalized small-mask supervision; test held behind promotion gate

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
consistent gain retain j01 and continue targeted optimization, WITHOUT test.
The user's newer instruction supersedes immediate post-round export: only
meaningful reproducible improvement may trigger test, per NEXT_SUBMISSION_PLAN.md.
Gate before outcomes: final APs gain>=.01,>=2 positive trained checkpoints,
and gain>=.01 repeated on unused-scene confirmation and a second matched seed.
This is a practical effect-size/reproducibility gate, not statistical significance.

Preflight completed on c2f0ead source (model code unchanged from32cf927):
-108 CPU regression tests PASSED.
-4 real-image GPU comparisons (two images x two candidates) exactly match
  original masks/scores/classes/boxes. Both candidates have46,072,854 parameters,
  identical to retained P2; neither adds inference parameters.
-Both10-update train-only smokes PASSED, including finite/nonzero interior P2
  gradients and strict final checkpoint evaluation/reload. At update10 the batch
  had35 retained targets and3 originally-small instances; binary weights1..2,
  continuous weights1..1.26729393, normalized all-target means1.08571434 /
  1.01330614. Reweighting therefore exercised real small training GT.
-Artifacts: runs/screen_v7_gpu_audit.json, runs/screen_v7_preflight/status.json,
  runs/screen_v7_smoke_{candidate}/instance_weight_step10.json.
Smoke uses2 fixed val images as execution checks only; scores are not selection
evidence. No candidate APs improvement is claimed before formal screening.
