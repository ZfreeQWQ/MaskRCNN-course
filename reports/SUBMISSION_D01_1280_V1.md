# First Kaggle submission: frozen geometry d01 at1280

User explicitly requested one submission to Kaggle competition rgb-d.
Live Overview/Evaluation and upload dialog checked2026-10-25:
small-mask AP@[.50:.95], CSV image_id/PredictionString, JSON instances with
category/confidence/compressed COCO RLE,1024-square masks,at most100/image,
2112 rows+header. Dataset README's old box-only instructions are obsolete.

Select existing full-val-supported configuration, not by looking at test:
d01 geometry model30k, inference1280, full2584 val APs.5079599314887093,
overall AP.8065509241755314. No Kaggle score is known yet.
weights runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth
SHA256 f33d2d37c422d6624d0e77bcb7843d4c7ca5af190241ecf1f02a24458ba8a11b.
Use configs/v4/h00_fast1280.yaml: exact-value-equivalent compact preprocessing,
no new weights, TTA, threshold search, training, full-val rerun or test tuning.

## Export correctness fixes

Old export_submission.py used RGBDMapper unconditionally, which would supply
4 channels to the11-channel geometry model. It now uses the same configured
mapper as validation, strictly loads all trained weights, records split=test,
and checks actual1280 input plus1024 output. Original upstream model code is
unchanged. State-dict version metadata is preserved; no checkpoint rewrite.

test_images.json provides no K. All14432 train and2584 val images share exactly
one K: fx=fy1910.81005859375,cx=cy511.5. Export requires the explicit
--assume-constant-training-intrinsics flag, checks train/val equality, then
adds test-only in-memory camera entries using this training constant. This is
an assumption about the unseen test camera, NOT supplied test intrinsics.
No train/val camera cache or train-only geometry normalization is modified.
Per-image geometry/RANSAC still uses only supplied test depth and image ID.

## Execution and validation

76 tests passed, including4 new contract tests and prior72 regressions.
One actual test-image GPU smoke passed strict load, input/output size checks,
JSON/RLE encoding and pixel decode roundtrip. This smoke is never uploaded.
run_submission.py freezes a clean Git snapshot and runs two disjoint index
shards on otherwise idle GPUs. Same model/scale for all2112 images.
No score cutoff beyond0, matching validation; no labels are available on test.
Every row/score/RLE area and exact metadata-ID order is validated on merge;
full decode roundtrip checks every100th image. Existing outputs are never
overwritten. Per-shard configs/camera assumption/checkpoint SHA and final
CSV SHA recorded under runs/submission_d01_1280_v1.

Uploading is a separate final step. Do not claim submitted or a test score
until Kaggle confirms the actual submission/result.

## Export result and upload blocker

Frozen source c20db7e6504f361f74e1a64dbd962abac1c615c7 completed both1056-image
shards and merge checks. Final CSV has2112 rows,211200 instances,max100/image,
68,782,875 bytes. SHA256:
d1e1803e6bc4c58128ccd393262fa219b7506110b3394778820cc7b95ddb64a3.
Local downloaded copy has the exact same size/SHA, under the chat workspace
outputs/kaggle_d01_1280_v1/submission.csv.

Browser is signed in to Kaggle as MagicGiraffe; description entered, but
filechooser.setFiles was rejected because the ChatGPT Edge extension lacks
Allow access to file URLs. No file was uploaded and Submit was NOT clicked.
No Kaggle score exists. Do not retry automatically or change extension access
without the user's action-time permission. User can upload this existing CSV
manually, or enable the permission before asking to continue. Do not regenerate
test predictions or launch training to solve this browser-only blocker.
