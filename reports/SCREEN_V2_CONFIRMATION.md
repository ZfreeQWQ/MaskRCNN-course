# Independent-scene confirmation: P2 query direction provisionally retained

Confirmation source6ab8fcc; model source4312dc1. Exactly one winner chosen
before confirmation, final3000 checkpoint, no runner-up search. Existing reserved
200 views/100 scenes disjoint from direction screen; no training/full-val/test.

| Metric | Matched i00 control | P2 query j01 | Delta |
| --- | --- | --- | --- |
| Small APs | .4158053261 | .4241058479 | +.0083005218 |
| Overall maskAP | .7531488681 | .7563717288 | +.0032228607 |
| Small AP75 | .4489378638 | .4592971412 | +.0103592774 |
| Small AR | .5023545706 | .5112188366 | +.0088642659 |

Direction-screen final APs delta was+.0060307894 on another200 views. Same
positive sign repeats; this is not significance or full-val evidence. Absolute
APs differ between panels because the images differ, not because the model
regressed between evaluations. Do not compare the two panels as a training curve.

Diagnostics are independent best-IoU oracle coverage, not COCO recall:

- Tiny<=64:bestIoU>=.5 count7/56 candidate vs8/56 control;>=.75 only1/56 both.
- Sparse small:bestIoU>=.75 count76/249 candidate vs69/249 control.
- Elongated:bestIoU>=.5 count106/152 candidate vs114/152 control.

Decision: preserve j01 as a provisional low-parameter improvement, retaining
the original d01/full-val best and all checkpoints. Compact inference export:
`runs/screen_v2_confirmation/j01_p2_queries_inference.pth`, exactly equal to final
trained tensors and preserving OrderedDict module metadata. Do not silently
replace the original baseline or claim full-val1280 improvement.

Next: independently test thin-edge-preserving local relative geometry and a
normal/depth-validity-aware version from the same d01 task checkpoint, using
the original direction-screen panel and matched control. Do NOT reuse this
confirmation panel to tune new settings. Combine with P2 only after an isolated
gain is established, then use a new confirmation protocol as appropriate.
