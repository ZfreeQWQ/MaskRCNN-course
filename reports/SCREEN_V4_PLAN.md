# Screen V4: tiny-instance training visibility, two separate factor ablations

P2 j01 gain repeated on direction screen(+.00603) and independent-scene small
confirmation(+.00830), but tiny high-IoU still1/56. Early-attention dilation and
local geometry did not improve final smallAPs. Do not stack rejected branches.

## Candidates vs the SAME P2 architecture

- m01_p2_min_scale08: only training MIN_SCALE .5->.8, MAX_SCALE stays1.5.
  RGB/depth/geometry/masks still use the same synchronized transforms. Fixed
 1024 crop/input. Reduces harmful scale-down, no larger inference or extra data.
- m02_p2_points32768: only official TRAIN_NUM_POINTS12544->32768, for original
  Hungarian matcher and mask losses. Same sampling rules and loss formulas,
  no GT-driven proposals/ROI points/new loss weights. No inference cost change.
- No new architecture parameters beyond the already tested P2 initializer.

The point hypothesis is narrower than claiming a diagnosed bug. A spatially
uniform matcher set can miss tiny targets. For an idealized areaA on1024^2,
the probability of at least one sample is1-(1-A/1024^2)^N. This is only a
sampling proxy: bilinear mask sampling, transformed areas and auxiliary losses
make actual behavior more complex, and stride4 precision remains a limitation.

Both start at the SAME d01 task checkpoint with identical zero-initialized P2
branch, not from the33k winner's weights. Same+3000 updates, batch2 FP32, seed42,
optimizer/shared and new-module LR, schedule, dataset/sampler,200 screen images.
Compare against completed j01 P2 control(final smallAPs.4705915273), not i00.

Explicit protocol differences are audited per candidate. Only INPUT.MIN_SCALE
and MODEL.MASK_FORMER.TRAIN_NUM_POINTS may differ; split, seed, LR, resolution,
loss opt-ins and all other shared settings remain guarded. Same seed is NOT a
claim of identical subsequent CUDA RNG streams when point count changes. More
points also mean more training compute/memory; report time, not equal wall time.

Evaluate0/1000/2000/3000 only on original fixed screen. Do not revisit consumed
j01 confirmation panel or run full val/test. Any shortlisted new direction needs
a fresh predeclared confirmation design, not tuning on the old confirmation.

Entry: run_screen_v2.py --suite screen_v4 --candidates m01_p2_min_scale08
m02_p2_points32768 --gpu-audit runs/screen_v4_gpu_audit.json
--smoke-prefix screen_v4_smoke_ --control-dir runs/screen_v2/j01_p2_queries
--allowed-change INPUT.MIN_SCALE MODEL.MASK_FORMER.TRAIN_NUM_POINTS

## Train-only visibility audit / preflight

512 evenly spaced records from train index (linspace0..14431), original
noncrowd areas only:1707 small instances,204 area<=64,320 area65..256.
No confirmation or test data used. Original areas are not transformed/cropped
visibility measurements; this audit motivates a hypothesis rather than proving it.

Idealized uniform-point probability of any hit on raw1024^2 canvas:

| Area | N12544 | N32768 |
| --- | --- | --- |
| 1 | .01189 | .03077 |
| 16 | .17420 | .39347 |
| 64 | .53497 | .86467 |
| 256 | .95325 | .99966 |

New budget is2.61x more points, NOT a claim of2.61x training latency. It still
does not solve1-pixel mask precision. Both candidate initial inference outputs
must remain exactly baseline. Strict transfer and staged P2 gradients are
checked by10-step real training before formal launch. A regression also checks
the number reaches the real official matcher AND criterion, not just config text.

Preflight passed72 tests,4 exact GPU initial comparisons, and both10-step real
training/reload checks with finite nonzero inner P2 gradients. Smoke workers=0
and roughly1.2s data loading dominates its step time; do not use those few-step
timings as steady-state GPU speed. Formal workers=4 and actual timing is recorded.
Both models retain46,072,854 parameters; these are training-only factors.
Scale changes can also alter retained GT counts and hence sampling RNG consumption;
neither candidate claims paired-identical CUDA draws, only identical seed/sampler.
