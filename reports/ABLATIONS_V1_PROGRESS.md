# Ablation progress

Observed: 2026-10-23T14:06:07.671803+08:00

Immutable training source: `28b288df6c49e25e489b0950eaf04829913e7864`. All scores 0–1.

Only final 30k checkpoints on all 2,584 validation images appear below.

| Experiment | Status | APS | Delta APS | Mask AP | Delta AP |
| --- | --- | --- | --- | --- | --- |
| baseline | completed | 0.291624 | 0 | 0.634357 | 0 |
| e01_small_resample | completed | 0.294290 | +0.002665 | 0.634965 | +0.000608 |
| e02_small_crop | completed | 0.294570 | +0.002946 | 0.634862 | +0.000505 |
| e03_rgbd_copy_paste | completed | 0.292533 | +0.000908 | 0.635833 | +0.001476 |
| e04_small_loss | completed | 0.297193 | +0.005568 | 0.629164 | -0.005194 |
| e05_foreground_points | completed | 0.192724 | -0.098900 | 0.604466 | -0.029891 |
| e06_depth_p2_gate | training (12840/30000) | — | — | — | — |
| e07_resolution1280 | pending (0/30000) | — | — | — | — |

All completed runs passed source-hash, COCO-transfer and split-use audits.
Training-only changes do not imply faster inference; small timing differences need controlled repeats.
No combinations have been tested. Record trade-offs and negative results.
These are single-seed observations, not statistical significance claims.
