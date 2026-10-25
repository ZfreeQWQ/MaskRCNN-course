# Screen V1 result: do not promote early attention dilation

Completed2026-10-25 17:18 Asia/Shanghai; total35min45s. Source423ad21.
Fixed200 val views /100 scenes,1024 inference, same d01 task checkpoint,
fresh AdamW,+3000 updates in each arm. No full val, no confirmation or test use.

| Added updates | Control APs | Early dilation APs | Paired delta |
| --- | --- | --- | --- |
| 0 | .457486 | .449329 | -.008157 |
| 1000 | .452086 | .449882 | -.002204 |
| 2000 | .454735 | .464559 | +.009824 |
| 3000 | .464561 | .462146 | -.002414 |

Final overall AP: control.785908 vs dilation.781939. Intermediate positive
ranking did not persist: not a reproducible improvement. Do not select the
2000 checkpoint merely because it beats control at that step. Keep checkpoints
and negative result. Frozen d01 and prior full-val reference remain unchanged.

Oracle tiny(area<=64) best-IoU>=.5 counts:11/68 control vs14/68 dilation;
>=.75 is1/68 in both. This is NOT COCO recall. Sparse small best-IoU>=.75
is100/314 control vs90/314 dilation. Broad expansion may introduce interference
without fixing precision; this explanation is a hypothesis, not a causal proof.

Next priority: retain original attention, add zero-initialized image-conditioned
query residuals from P2 mask features, with an explicit geometry variant.
