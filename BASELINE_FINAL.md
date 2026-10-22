# Frozen RGB-D baseline — 2026-10-22

Mask2Former + ResNet-50, official upstream commit
`9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`, four-channel early fusion.
Official network sources are unchanged; course data/training/evaluation are adapters.

Run: `/workspace/Mask2Former-course/runs/baseline_r50_rgbd/`.
Checkpoint: `model_final.pth`; final evaluation: `val_full_final.json`.
Completed 30,000 updates, batch 2, FP32, seed 42, AdamW 1e-4,
backbone multiplier 0.1, LR drops at 18k/24k. Training: 3h25m;
final full-val evaluation: 25m25s. Train 14,432 / val 2,584 images.
Train and val never merged; test not used for training or selection.

| Completed updates | Full-val mask AP | Full-val small-mask APS |
| --- | --- | --- |
| 5,000 | 0.5659242191 | 0.1839579556 |
| 10,000 | 0.5722585062 | 0.2303435207 |
| 15,000 | 0.6027394262 | 0.2324246500 |
| 20,000 | 0.6250504965 | 0.2784973044 |
| 25,000 | 0.6338306057 | 0.2895472546 |
| **30,000 (fixed final)** | **0.6343571918** | **0.2916244869** |

Final AP50 0.8595655265, AP75 0.7009566463, APM 0.6627377348,
APL 0.8514475691. All scores 0–1, COCO segm IoU .50:.05:.95,
maxDets 100, original 1024x1024 masks. Small uses official COCO area range.

Inference 0.0643364 s/image on L20, warmed up, excluding disk/RLE/COCO
aggregation. 43,953,090 network parameters. Training checkpoint includes
optimizer (~528 MB); do not report that as inference-only model size.

The last 5k updates show a plateau; this is a usable single-seed baseline,
not a proof of global convergence or cross-seed statistical significance.
The initial val400/1k monitoring score is not used for final comparisons.

All 555 transferred COCO tensors matched at initialization. RGB kernels
copied exactly, added depth kernels initialized zero. Nonzero finite depth
gradients confirmed; final depth stem norm 0.05456005. This establishes
connectivity, not the causal gain of depth (requires RGB-only ablation).

Retired Mask R-CNN artifacts remain recoverable outside the active project:
`/workspace/.trash/retired-maskrcnn-20261021-5hlhuwlo/`.
Supplied data and Git history are preserved. No TensorFlow is used.
