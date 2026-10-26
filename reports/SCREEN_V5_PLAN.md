# Screen V5: isolate small-instance positive sampling floors

Evidence: V4 increased uniform budget did not improve final APs, while actual
train-only coverage audit found29/44 retained tiny GTs had no positive matcher
point under12544. Prior large-fraction foreground loss and per-GT ROI costs
were negative. Do not repeat or stack those changes.

## Two isolated candidates

- n01_p2_match_floor4: matcher only. Retain12544 points per image, replace the
  tail with up to4 positive pixel-center coordinates per retained small GT.
  Round-robin GTs; total cap256(<2.05% of budget). ALL queries AND ALL GT masks
  use that same common coordinate set. Same class/BCE/Dice cost weights and
  formulas, no per-column ROI objective or extra GT-specific cost mixing.
  If there are more than64 small GTs, cap reduces anchors per GT; first anchor
  per GT retained whenever small-count<=256. Dataset has100 query/output slots.
- n02_p2_loss_floor4: loss only. Original uniform Hungarian matcher unchanged.
  Replace exactly4 coordinates in the original RANDOM tail of each matched
  retained small GT's12544 uncertainty/random loss sample(<.032%). Keep
  uncertainty-selected coordinates, all nonsmall matches, original BCE/Dice
  formulas, weights and instance normalization. Not old10% foreground sampling.
- Do not combine these directions in this round. Model/P2/geometry parameters
  and inference remain exactly unchanged, no extra inference cost.

Small flags come from original training-instance area<=1024. Positive anchors
come from each GT's actual augmented/padded mask, not bbox union: deterministic
raster-order foreground quantiles at pixel centers. One-pixel GT gets repeated
coordinates. This is a deliberate tiny-instance sampling bias, NOT an unbiased
estimator of global BCE/Dice and NOT a claim of guaranteed AP improvement.

Anchors use no RNG. Full original uniform/uncertainty point samplers still run
with identical shapes before replacements. Off modes delegate exactly upstream;
enabled unit tests check matching/loss RNG state equality. Real assignment
differences change the optimization trajectory; no numerical identity claim.
GT anchors are created in prepare_targets for supervised training ONLY, never
in inference. No annotation, pose/model ID or GT query location is supplied
to the inference model. No official Mask2Former/Detectron2 file changes.

## Budget and acceptance

Same d01 trained30k task checkpoint, zero-initialized P2 module, fresh AdamW,
seed42, batch2FP32, shared LR1e-5/backbone1e-6/new P2 LR1e-4,1024 input,
flip/scale.5..1.5/crop,3000 extra updates,schedule2000/2600,warm100.
Compare final3000 against existing j01P2 .4705915273, not i00 and not30k frozen.
Same fixed200 direction panel evaluated0/1000/2000/3000. Check APs primary,
small AP75/AR, overall AP and tiny/sparse oracle diagnostics. Do not select
single middle peak. No full val or test use. Old P2 confirmation panel consumed:
do not retune on it or call it unseen; a consistent new winner needs a fresh
predeclared independent-scene check before stronger claims.

Narrow protocol differences explicitly allow MODEL.META_ARCHITECTURE registration
wrapper and COURSE.COVERAGE.MATCH_POINTS or LOSS_POINTS only. Legacy protocols
with absent COVERAGE mean disabled defaults, never arbitrary relaxed settings.
Replacement cap, all old loss opt-ins, data/seed/LR/augmentation remain guarded.

Preflight: unit/off-mode parity, shared init/RNG/strict load, GPU exact initial
predictions, two10-step real train smokes with finite gradients and strict
checkpoint reload. Launch only from clean versioned source snapshot on idle
GPUs. Preserve all best weights/logs and failed preflight artifacts.

## Passed preflight

87 CPU regression tests passed, including8 coverage tests. Four real GPU
initial comparisons (two images per candidate) had exact masks/scores/classes/
boxes, with46,072,854 parameters for either candidate. Both10-step real training
smokes passed finite losses, nonzero inner P2 gradients and strict final reload.
An audit-enhanced repeat (separate r2 output folders) confirmed the floor was
actually exercised: at step10 each arm processed3 small targets /12 replacement
coordinates in its selected path. Matcher arm's loss floor stayedoff; loss
arm's matcher floor stayedoff. Step1 happened to contain no small targets, so
checking only its diagnostic would not establish enabled-path coverage.

Successful smoke paths: runs/screen_v5_smoke_r2_n01_p2_match_floor4 and
runs/screen_v5_smoke_r2_n02_p2_loss_floor4. Functional2-image smoke scores are
NOT optimization evidence and are never compared for candidate selection.

Formal entry:
run_screen_v2.py --suite screen_v5
--candidates n01_p2_match_floor4 n02_p2_loss_floor4
--gpu-audit runs/screen_v5_gpu_audit.json --smoke-prefix screen_v5_smoke_r2_
--control-dir runs/screen_v2/j01_p2_queries
--allowed-change MODEL.META_ARCHITECTURE COURSE.COVERAGE.MATCH_POINTS COURSE.COVERAGE.LOSS_POINTS
