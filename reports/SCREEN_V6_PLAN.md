# Screen V6: query-conditioned nonlinear point refinement

Motivation: V3 local geometry and V4/V5 sampling changes failed final APs.
Existing P2 screen has208/921 small GT with bestIoU.5..75, including165/746
area257..1024. See DATA_ERROR_AUDIT.md for oracle caveats. Do not claim purely
boundary-caused errors, guaranteed gains or recovery of wholly missed targets.

Primary reference: PointRend https://arxiv.org/abs/1912.08193 (read abstract).
This is PointRend-INSPIRED, NOT a reproduction/SOTA claim: nonlinear pointwise
mask prediction and adaptive uncertain-location subdivisions, adapted to
Mask2Former queries and supplied RGB-D. No new data or upstream-file edits.

## Distinction from existing V3 g02

g02 used dense low-rank query x stride2 feature residuals and gained only
.00191 full-val APs at1024. This is NOT simply rerunning that refiner: each
point MLP consumes fine features, instance mask embedding and its sampled
coarse logit jointly, so correction is nonlinear and query/position dependent.
Training uses point computations only, inference adaptively updates uncertain
pixels. Retain proven P2 query initializer; do not add rejected coverage floors.

- o01_p2_point_rgbd: pretrained4-channel RGB-D stem BEFORE maxpool,project16ch,
  3x3 fine features,16ch query projection,33->64->64->1 point MLP.
- o02_p2_point_geometry: same RGB/query/MLP weights at initialization, plus
 7->16 explicit-geometry projection added to fine RGB-D features. Multiply by
 binary valid-depth and reuse existing whole-branch dropout mask, no bypass.
- Bounded logit residual tanh*1, last layer zero-init. Original coarse
  predictions, learned queries, P2 initialization, attention and auxiliaries
  unchanged. No residual branch feedback into attention or matching.
- Original Hungarian matching on coarse masks. All auxiliary losses unchanged.
  Final loss uses original12544 uncertainty/random points sampled on coarse
  masks, original BCE/Dice/normalization, but logits=coarse+point correction.
  No GT-driven sample points, area weights, new boundary loss, extra point RNG.
- Inference: two subdivisions to stride2 and full input size,2048 uncertain
  points/query/stage,query chunks16. Original coarse field is interpolated
  DIRECTLY to each grid; only residual is interpolated between grids. This
  prevents accidental double-bilinear baseline changes at zero initialization.
  Point MLP always conditions on original coarse values, matching training.
- Single registered parameter owner sem_seg_head.point_refiner; criterion
  holds a weak reference. Rebuild from config+state_dict for export/reload,
  never pickle/deepcopy a live model object. Preserve OrderedDict._metadata.

## Protocol and decision

Same d01 task30k weights, new P2/point modules zero-output, fresh AdamW,+3000
updates,batch2FP32,seed42,input1024,originalflip/scale.5..1.5/crop,12544 points.
Base LR1e-5/backbone1e-6/P2+point modules1e-4,steps2000/2600,warm100.
Compare final3k to j01P2 final .4705915273, not frozen30k or i00. Same200 panel
at0/1000/2000/3000. Main APs + AP75/AR/overall/tiny/sparse controls, no midpoint
peak selection. Inference and training cost can increase; measure, don't claim
zero overhead. No full val,test,scale sweep or reused old confirmation panel.
Consistent winners need new predeclared independent scenes before stronger claims.

Architecture class changes and POINT module settings explicitly documented;
shared data/seed/LR/loss opt-ins guarded by compare_protocol, only meta wrapper
exception. Preflight requires CPU forward/loss/gradient/RNG/metadata tests,
GPU exact initial predictions and10-step real gradients/strict reload.

Formal entry run_screen_v2.py --suite screen_v6
--candidates o01_p2_point_rgbd o02_p2_point_geometry
--gpu-audit runs/screen_v6_gpu_audit.json --smoke-prefix screen_v6_smoke_
--control-dir runs/screen_v2/j01_p2_queries --allowed-change MODEL.META_ARCHITECTURE

## Passed preflight

96 regressions passed,9 specifically cover point decoding/refinement/loss RNG,
native-coarse identity, gradients, chunk consistency, independent images,
geometry validity/common initialization, shared backbone/dropout/LR, strict
state-dict roundtrip/module metadata and single parameter ownership.
GPU audit:4 real comparisons(two images x two candidates), exact initial
masks/scores/classes/boxes.633 trained checkpoint tensors transfer exactly;
allowed new state only under P2 initializer and point_refiner prefixes.
Two10-step real training/reload smokes passed. The output layer gets gradients
at step1; interior fine/query/MLP and optional geometry gradients are finite
and nonzero by10. Tiny first10-step inner gradients are expected from zero
output initialization, not evidence of final convergence or AP improvement.
Sample smoke loss rendered439040 point predictions, preserving12544/match.

Parameters:46086759 RGB-D /46086903 geometry vs46072854 P2 control: +13905 /
+14049. Optional geometry adds144 parameters; all common point-head parameters
have identical initialization between candidates. Model size alone does not
predict inference overhead. Log residual mean/std/RMS during training to help
distinguish mostly constant mask-logit calibration from local corrections;
this scalar diagnostic alone does not establish causal boundary improvements.

Smoke folders runs/screen_v6_smoke_o01_p2_point_rgbd and
runs/screen_v6_smoke_o02_p2_point_geometry. Two-image smoke metrics are functional
checks, never candidate-selection scores. No test images were read in preflight.
