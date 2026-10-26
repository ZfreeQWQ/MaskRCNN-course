# V7: neither small-loss weighting improves primary APs

Source703a8f9; both finished+3000 updates from d01 trained30k weights with fresh
P2 initialization/optimizer, seed42, same data/points/augmentation/frozen200.
No test, full val or independent confirmation. No new inference parameters.

| Added updates | P2 control | Small x2 | Continuous area weight |
| --- | --- | --- | --- |
|0|.4574859531|.4574859531|.4574859531|
|1000|.4561408882|.4594932334|.4547307360|
|2000|.4699739763|.4672065477|.4559884836|
|3000|.4705915273|.4696708076|.4644576403|

Final deltas -.0009207198 / -.0061338870; reject both for promotion. x2 is nearly
flat, not a meaningful improvement and does not justify independent/test use.
Its early positive does not persist. Continuous weighting negative throughout.

| Final metric | P2 control | Small x2 | Continuous |
| --- | --- | --- | --- |
|Overall AP|.7855258200|.7858177785|.7877183596|
|Small AP75|.5062965451|.5114183645|.4992012660|
|Small AR|.5193268187|.5166123779|.5097719870|
|Tiny bestIoU>=.5|14/68|14/68|7/68|
|Tiny bestIoU>=.75|1/68|1/68|2/68|
|Low-fill bestIoU>=.75|96/314|100/314|98/314|
|Area257..1024 bestIoU>=.75|487/746|493/746|479/746|

Overlap counts are oracle diagnostics, NOT COCO recall. x2 modestly improves
AP75 and low-fill overlap but not multi-threshold APs/AR. Do not promote a
secondary metric. Loss weights really affected small train instances and both
branches had finite/nonzero gradients, so this is not an inactive factor.

Next: training resolution. Existing d01 inference-only1280 gain is established
but training at1280 has not been completed. Compare train1024 vs1280 P2 with
BOTH evaluated at1280 on the same fixed200, reusing old control checkpoints but
re-evaluating them at1280. Never credit a1024-vs1280 inference difference to
training. Maintain the >=.01 plus independent/second-seed gate; no test now.
