# Screen V2: modest P2 query gain; independent confirmation required

Source4312dc1. Two new candidates, same d01 checkpoint/+3000updates, original
attention and losses,1024,batch2 FP32,seed42. Matched i00 control reused only
after config/weights/RNG/initial exact CPU/GPU parity and geometry/panel checks.
All rows below are the frozen200-image/100-scene direction screen, NOT full val.

| Added updates | Control APs | P2 queries APs | P2+explicit geometry APs |
| --- | --- | --- | --- |
| 0 | .457486 | .457486 | .457486 |
| 1000 | .452086 | .456141 | .453223 |
| 2000 | .454735 | .469974 | .467245 |
| 3000 | .464561 | .470592 | .467257 |

Final paired APs delta: P2+.0060307894; explicit geometry+.0026966152.
P2 final overallAP.7855258200 vs control.7859084907 (slightly lower),
small AP75.5062965451 vs.5078363472 (slightly lower),
smallAR.5193268187 vs.5100977199. This is not a broad precision improvement.

Independent best-IoU diagnostics (oracle coverage, NOT COCO recall):

- Tiny<=64 bestIoU>=.5:14/68 P2 vs11/68 control, but>=.75 only1/68 in both.
- Sparse small bestIoU>=.75:96/314 P2 vs100/314 control. Broad fragment-boundary
  improvement is not demonstrated. New query conditioning mostly does not fix
  the sub-P2-resolution/tiny-mask precision bottleneck.
- P2 mask features already contain the existing four-scale geometry fusion.
  Additional explicit geometry is NOT the same as adding depth to an RGB model;
  no evidence here that duplicating depth input improves query conditioning.

## Decision before viewing confirmation

Select ONLY j01_p2_queries final3000 checkpoint: highest final APs among candidates
positive at>=2 trained checkpoints (j01 positive at all3). No lucky intermediate
checkpoint selection. Evaluate against i00 final3000 on the existing untouched
panel_confirmation.json:200 images/100 disjoint scenes,722 small/56 tiny GT.
Exactly one confirmation comparison; do not try the runner-up or altered settings
on this panel if selected candidate fails. No training or full-val/test use.

If gain repeats, retain as a provisional low-overhead improvement, not statistically
proven/SOTA/full-val gain. Continue model work focused on local geometry reliability
and high-resolution mask precision, since tiny high-IoU and sparse boundaries
remain weak. If not repeated, keep trained variants for audit and do not promote.
