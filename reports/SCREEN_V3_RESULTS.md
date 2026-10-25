# Screen V3: reject unfiltered contrast; reliability recovers control, no final gain

Source0117934. Same d01 task checkpoint, frozen200 screen views,+3000 updates,
1024,batch2 FP32,seed42. Original matching/loss; extra local modules zero-init,
68 regressions and real GPU parity/10-step gradients passed. No confirmation/test.

| Added updates | i00 control APs | Local contrast APs | Reliable contrast APs |
| --- | --- | --- | --- |
| 0 | .457486 | .457486 | .457486 |
| 1000 | .452086 | .451255 | .459064 |
| 2000 | .454735 | .454501 | .468091 |
| 3000 | .464561 | .457906 | .464507 |

Final deltas:-.0066542880 unfiltered,-.0000537719 reliable. The apparent
positive reliable signal at1000/2000 did not persist versus matched final control.
Do NOT promote a lucky intermediate checkpoint or spend a confirmation panel.

Reliable overallAP.787836 vs control.785908, but primary smallAPs does not improve,
smallAP75.501832 vs.507836, smallAR.507275 vs.510098. Task objective, not overall
AP, determines promotion. Tiny>=.75 remains1/68 in reliable and control; sparse
>=.75 is91/314 reliable vs100/314 control (oracle diagnostics, NOT COCO recall).

Decision: retain P2 query j01 from the independently confirmed prior experiment;
do not combine these local-geometry branches with it. Keep code/checkpoints as
negative-result audit, not required baseline components.

Next hypotheses address training visibility rather than add more branches:
reduce the scale-down augmentation that further shrinks already tiny targets;
increase official matcher/mask-loss point budget to reduce coverage/sampling noise.
Run as two separate single-factor changes to the P2 model, using its matched
completed j01 control, not the weaker i00 control. No architecture/loss-formula or
GT-input changes. Point coverage is a hypothesis, not a proven cause of errors.
