# Screen V5: reject both sparse positive-coverage floors

Source ace63b5. Both candidates finished3000 additional updates from the same
d01 checkpoint with fresh P2 initialization, original fixed200 screen and same
optimizer/seed/augmentation. Coarse matching-only or final+aux loss-only4-point
floor, never both. No test/full-val/confirmation was used.

| Added updates | P2 j01 control APs | Matcher floor4 | Loss floor4 |
| --- | --- | --- | --- |
|0|.4574859531|.4574859531|.4574859531|
|1000|.4561408882|.4549655032|.4575805905|
|2000|.4699739763|.4578518053|.4611477978|
|3000|.4705915273|.4639842516|.4639028224|

Final deltas -.0066072757 / -.0066887050. Loss floor's small1000-step positive
did not persist. Do not promote intermediate peaks or spend independent scenes.

| Final metric | P2 control | Matcher floor | Loss floor |
| --- | --- | --- | --- |
|overall AP|.7855258200|.7869033081|.7864944973|
|small AP75|.5062965451|.4988943410|.4971555091|
|small AR|.5193268187|.5105320304|.5100977199|
|tiny bestIoU>=.5|14/68|13/68|14/68|
|tiny bestIoU>=.75|1/68|1/68|1/68|
|sparse bestIoU>=.75|96/314|95/314|92/314|

Oracle overlap counts are not COCO recall/AP. Both precision and recall controls
also deteriorated. Retain existing P2 model, do not combine rejected floors.
Finite losses, real positive-coverage audits and gradients were checked, so
this is a measured negative direction, not an unexercised/no-op experiment.
It does not prove all targeted supervision is useless, only this short-budget
setting is not worth stacking. Training itself: ~.4911/.4869 s per update.

Next: change the final mask prediction representation using query-conditioned
nonlinear point residuals. Preserve matching, auxiliary losses, original point
distribution and baseline interpolation. See DATA_ERROR_AUDIT.md and SCREEN_V6_PLAN.md.
