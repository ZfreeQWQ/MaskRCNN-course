# Historical running status — superseded by BASELINE_FINAL.md

The baseline has now completed. Final full-val AP is 0.6343571918 and
APS is 0.2916244869. The text below is the archived initial progress note,
not current status. See [BASELINE_FINAL.md](BASELINE_FINAL.md).

The active model is **Mask2Former + ResNet-50**, with RGB-D four-channel
early fusion. No retired model code or weights are used by this run.

## Completed

- Removed the retired checkout, weights and logs from the active workspace,
  recoverably. The remote archive is
  `/workspace/.trash/retired-maskrcnn-20261021-5hlhuwlo/` (~13 GB).
- Local retired scripts/artifacts were moved to
  `C:/Users/86133/Documents/Codex/archives/retired-maskrcnn-20261021/`.
- Preserved the supplied dataset, task instructions, test metadata and Git
  history. The new remote working directory is
  `/workspace/Mask2Former-course`, still on `wzh-dev`.
- Installed official Mask2Former/Detectron2 native CUDA extensions for L20.
- Downloaded official COCO instance-segmentation R50 weights; SHA256:
  `c18e6551f9c0e951e7ee3a63eb5402b901612e4e141965d4719d13c69a97ec2a`.
- Verified all **555 retained weight tensors** match after loading.
- Verified finite, nonzero depth-channel gradients.
- Four data tests passed: depth range, synchronized geometry, zero padding,
  and disjoint train/val IDs. Ten-update smoke training/checkpoint/inference
  passed; its 2-image scores are not baseline measurements.

## Formal run

`/workspace/Mask2Former-course/runs/baseline_r50_rgbd/`

PyTorch 2.5.1 + CUDA 12.4, Python 3.11.10, batch 2, FP32, seed 42,
AdamW base LR 1e-4, backbone multiplier 0.1, 30000 updates. See README/config
for all settings. Approximately 0.4 seconds/update after warm-up; total
training is provisionally 3–4 hours, not a convergence guarantee.

GPU 0 trains, GPU 1 evaluates asynchronously. Step 1000 monitors fixed val400;
steps 5000/10000/... and final checkpoint evaluate all 2584 val images.
Only the final full-val result is the fixed-budget baseline report. Val is
never used for training, and test is not used for tuning.

The final baseline is **not available yet**. Do not confuse the initial
val400 monitoring result with a fully trained, full-validation baseline.

### First measured task checkpoint: 1000 updates, val400

| Metric (0–1) | Value |
| --- | --- |
| Mask AP@[.50:.95] | 0.5345480863 |
| Mask AP50 | 0.7992446887 |
| Mask AP75 | 0.5676437723 |
| Small-mask APS@[.50:.95], primary | 0.1856480000 |
| Medium AP | 0.5354265160 |
| Large AP | 0.8584445867 |

Checkpoint: `runs/baseline_r50_rgbd/model_0000999.pth` (1000 completed
updates; Detectron2 stores the iteration index starting at zero).
Forward/pre-postprocess time after warm-up: 0.06338 s/image, excluding file
reading and RLE encoding. Total val400 inference/encoding/COCO aggregation:
240.7 seconds. These are different timing scopes; do not present 0.06338 as
end-to-end submission generation latency. The training checkpoint contains
AdamW states (~528 MB), not just inference weights; use the weight-only
export utility for model size.

This measurement is a real validation result, not the 2-image smoke test.
It is still only a sampled, short-training checkpoint; final reporting
requires the final checkpoint and the complete validation split.

TensorFlow is not part of the new training runtime. Console/JSON logging
avoids importing TensorBoard's unrelated TensorFlow compatibility layer.
