# Screen V6: reject nonlinear point refinement under this screening budget

Both arms completed 3000 additional updates, source efa4141. Same trained d01
weights, P2 query module, fresh optimizer, seed, augmentation, losses, frozen
200-image screen and original matching/point coordinates as j01 control.
No test, full-val or independent confirmation used.

| Added updates | P2 control | RGB-D point head | + explicit geometry |
| --- | --- | --- | --- |
| 0 | .4574859531 | .4574859531 | .4574859531 |
| 1000 | .4561408882 | .4600664402 | .4558935658 |
| 2000 | .4699739763 | .4594288349 | .4580672550 |
| 3000 | .4705915273 | .4660299140 | .4651758808 |

Final APs deltas -.0045616134 / -.0054156466; no promotion based on the RGB-D
arm's early positive. Final small AP75 .5034921213 / .5035739716 vs .5062965451;
small AR .5131378936 / .5098805646 vs .5193268187. Overall AP .7875887571 /
.7856947043 vs .7855258200 does not compensate for the primary metric loss.
96 regression tests, GPU initial equivalence, finite losses and nonzero interior
point-head gradients passed; this was not a disconnected/no-op branch.

Retain previous P2 and full-val best weights/logs. This is a short-adaptation
negative result, not proof all point refiners or longer training fail.
Next action is a fixed-screen error audit with persisted unchanged predictions,
COCO one-to-one matches, predeclared density strata and a clearly label-informed
ranking probe. Do not infer a density/duplicate bottleneck from annotation
counts or independent best-IoU aggregates alone.
