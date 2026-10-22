# RGB-D Mask2Former course baseline

Active project: `/workspace/Mask2Former-course`, branch `wzh-dev`.
This replaces the retired model in the working tree while retaining Git history.

## Model and initialization

- Official facebookresearch/Mask2Former, commit
  `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`.
- ResNet-50 + multi-scale deformable pixel decoder + masked-attention Transformer
  decoder, 100 object queries, single component class + no-object.
- Official **COCO instance-segmentation** checkpoint `model_final_3c8ec9.pkl`.
  This is not a semantic/panoptic checkpoint or an unrelated detector.
- RGB-D baseline uses four-channel early fusion, **not gated fusion**.
  RGB stem kernels copy COCO exactly; the fourth depth kernel starts at zero,
  preserves initial RGB features, and receives gradients during training.
- Only the 80-class classifier is replaced; its class-dependent loss buffer
  is rebuilt. All 555 retained checkpoint tensors must match after loading.
- `prepare_weights.py` preserves state-dict version metadata and converts
  official historical `static_query` keys to `query_feat`. Without metadata,
  the upstream loader would incorrectly migrate the pixel-decoder keys.
  No upstream network computation or CUDA implementation was modified.

## Environment

`/opt/venvs/mask2former/bin/python`: Python 3.11.10, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, CUDA 12.4, cuDNN 9.1, NumPy 1.26.4.
Detectron2 and the official deformable-attention extension are compiled for
L20 compute capability 8.9. This isolated venv reuses the container's existing
PyTorch installation, not its retired TensorFlow environments.

## Data and evaluation

Only `/workspace/blender` is used. Train: 14432 images / 448469 instances;
val: 2584 images / 73421 instances. Val never participates in backpropagation;
test is reserved for inference. COCO initialization is explicitly permitted.
`data_cache/` is derived only from supplied train/val annotations: an indexed
SQLite database avoids repeatedly decompressing a full shard for each image.
Train/val IDs and paths are checked disjoint. Four loader workers are used.

RGB uses the official RGB channel order and normalization. Depth is uint16
times `depth_scale_m`, clipped at 1 meter and scaled to 0..255; invalid depth
is zero. Horizontal flip, scale 0.5..1.5, and 1024 crop/pad are synchronized
across RGB/depth/masks. RGB resize is bilinear; depth/masks are nearest-neighbor.
Mask/depth padding is zero, never foreground. **No Copy-Paste augmentation**.

Evaluation uses original 1024x1024 masks, standard pycocotools COCO `segm`
AP@[.50:.95], max 100 instances/image, and reports 0..1 scores. **APS is primary**.
The final fixed-budget checkpoint is the reported baseline; intermediate val
results only monitor progress. A short-run result is not a convergence claim.

## Fixed baseline recipe

Single L20, batch 2, FP32, seed 42, 30000 optimizer updates (60000 primary
image presentations, not 60000 unique images). Official AdamW parameter-group
rules, base LR 1e-4, backbone multiplier 0.1, weight decay 0.05, 250-step warmup,
LR drops at 18000 and 24000. This is **not an exact replication of the other
team's screenshot**: in particular their gated fusion/base LR 1e-5 differ.

GPU 0 trains; GPU 1 evaluates asynchronously, so training does not pause for
periodic validation. First checkpoint at step 1000 evaluates fixed val400;
steps 5000/10000/... and the final checkpoint use full val. Intermediate
evaluation may be skipped if the preceding one is still active. Final full
validation always completes before a run is marked completed.

```bash
cd /workspace/Mask2Former-course
/opt/venvs/mask2former/bin/python -m course.data --cache data_cache
/opt/venvs/mask2former/bin/python prepare_weights.py
/opt/venvs/mask2former/bin/python -m unittest -v test_rgbd_data
CUDA_VISIBLE_DEVICES=0 /opt/venvs/mask2former/bin/python train_rgbd.py \
  --output runs/baseline_r50_rgbd --eval-gpu 1
```

`--smoke-test --max-iter 10` checks training, exact weight transfer, depth
gradients, checkpoint save/load and two-image val inference. Smoke metrics
are not baseline results. Resume uses `--resume` with the same output path.
Source hashes, initialization/gradient audits, config, metrics and live status
are saved inside each run. Do not interpret `status=training` as completion.

`requirements-l20.txt` lists the required runtime only. The generated
`environment-observed.txt` records inherited container packages for audit;
it is NOT the installation requirements and may list unused packages.

Once a task checkpoint has been selected, `export_submission.py --weights
PATH --output submission.csv` supports the required test-only output:
`image_id,PredictionString`, JSON arrays with category_id/score/segmentation,
compressed COCO RLE at 1024x1024, <=100 instances, `[]` for no predictions.
This export utility is not invoked during baseline training or validation.

## Cleanup / recovery

The retired project's code, weights and logs were removed from the active
workspace via a recoverable move, not permanent erasure. Exact paths are in
`MIGRATION.json`; `/workspace/.trash/retired-maskrcnn-20261021-5hlhuwlo/`
contains the retired checkout and logs. Data, task instructions and Git
history were preserved. The GitHub repository's existing URL was not renamed.

## Upstream sources

- https://github.com/facebookresearch/Mask2Former
- https://github.com/facebookresearch/detectron2
- https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md
