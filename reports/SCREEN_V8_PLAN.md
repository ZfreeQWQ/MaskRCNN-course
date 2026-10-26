# V8: training-resolution ablation, common1280 inference

Motivation: INSTANCE_BOTTLENECK_AUDIT.md shows severe high-IoU/low-fill deficits.
V6 refiners and V7 reweighting did not raise APs. Existing d01 full-val inference
upscale1024->1280 raised APs .42415->.50796; that is NOT evidence of a training
resolution benefit. Old e07 training1280 was stopped before execution. Test a
direct representation-resolution change rather than another sampling tweak.

One candidate q01: INPUT.IMAGE_SIZE1280 (actual training resize/crop). Retain P2,
geometry, original points/loss/matcher, batch2 FP32, seed42, data split and scale
ratio. Same d01 weights, fresh P2/optimizer and+3000 updates; LR/decays unchanged.
All evaluation uses fixed1280 inference and original1024 masks/GT areas.

Control q00: existing j01 train1024 P2 initialized/1000/2000/final3000 checkpoints,
re-evaluated at1280 on the frozen200. Do NOT reuse old1024 APs numbers. Both arms
have same trained633 tensors and ALL initial P2 tensors, verified from old and
new initialized checkpoints including OrderedDict._metadata. This is equal
updates/image exposures, NOT equal FLOPs/pixel budget/wall time; report costs.
Control needs no repeated training; GPU0 re-evaluates its four saved checkpoints,
GPU1 trains/evaluates candidate synchronously. Physical evaluator locks remain.

Formal training requires explicit --training-size1280; old1024 default guard is
unchanged. Narrow protocol checks distinguish shared eval-size change from the
ONLY training-size difference. Initial1280 masks/scores/classes/boxes must match
GPU reference; real10-step training/strict reload must pass before formal run.
Fixed200 only,0/1000/2000/3000. No full val, test, new data or inference-scale sweep.
No V7 rejected losses/refiners/floors are stacked.

Promotion gate remains final same-scale APs delta>=.01 and>=2 positive trained
checkpoints, then independent unused-scene and second matched-seed replication
with delta>=.01. Practical reproducibility, not statistical significance. A score
above the old1024 .47059 alone is NOT evidence of a training improvement.

## Preflight passed, source90a04d8

113 CPU regressions passed. Actual train tensors on two supplied train records
are11x1024x1024 vs11x1280x1280, with the same retained38/40 targets and6/7
originally-small flags. Geometry statistics and original area labels unchanged.
Four actual1280 GPU comparisons exactly match initial masks/scores/classes/
boxes; model parameter count unchanged46,072,854. The real1280 ten-update smoke
passed with finite/nonzero inner P2 gradients and strict final checkpoint
evaluation/reload, without OOM. All643 initialized model tensors and module
version metadata exactly match the old trained1024 control's initial checkpoint.
Smoke scores are execution checks, not selection evidence. Smoke used0 data
workers, so its2.03s/update dominated by1.42s data time is NOT formal throughput.
Artifacts: runs/screen_v8_preflight.json, runs/screen_v8_data_audit.json,
runs/screen_v8_gpu_audit.json and runs/screen_v8_smoke_q01_p2_train1280/.
