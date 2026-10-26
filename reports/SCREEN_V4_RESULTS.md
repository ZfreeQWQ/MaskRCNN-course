# Screen V4: reject both isolated training-factor changes

Fixed200 direction panel, matched3k updates, same d01 checkpoint and P2 j01
architecture. Both factors explicitly declared; no test/full-val/confirmation.

| Added step | P2 control | min scale.8 |32768 uniform points |
| --- | --- | --- | --- |
|1000|.4561408882|.4501482955|.4549362231|
|2000|.4699739763|.4601322780|.4648413119|
|3000|.4705915273|.4634108138|.4650391553|

Final primary APs deltas -.0071807135 / -.0055523720. Reject both; never
stack them with the current P2 candidate just because overall AP rose slightly.

| Final metric | P2 control | min scale.8 |32768 points |
| --- | --- | --- | --- |
|overall AP|.7855258200|.7866787593|.7863010530|
|small AP75|.5062965451|.4986021400|.4936903437|
|small AR|.5193268187|.5114006515|.5163952226|
|tiny<=64 bestIoU>=.5|14/68|13/68|17/68|
|tiny<=64 bestIoU>=.75|1/68|2/68|2/68|
|sparse bestIoU>=.75|96/314|95/314|97/314|

Oracle best-IoU counts are NOT COCO recall/AP and allow predictions to be reused
across GTs. More uniform points improved this tiny diagnostic but failed the
primary metric and precision/recall controls. This does not prove sampling
coverage is irrelevant; train-only actual matcher audit still found many tiny
targets with no sampled foreground. It argues against merely increasing budget.

Retain P2 j01 model/weights and all original runs. Its +.00603 direction gain
and +.00830 independent-scene gain remain provisional, not full-val evidence.
