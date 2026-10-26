# User-authorized next submission after this optimization screen

Request 2026-10-26: finish the current optimization round, then produce a new
submission.csv. Do not indefinitely defer export by starting further screens.

1. Complete the next normalized-small-loss screen V7, compare fixed final3000
   against retained matched P2 control. Reject mid-run-only gains.
2. If no consistent benefit, retain existing P2 j01. If benefit repeats across
   trained checkpoints, use a predeclared unused-scene small confirmation
   against P2, never re-use the old P2 confirmation as untouched holdout.
3. Before export, compare selected P2 candidate and retained d01 submission
   recipe on the SAME frozen200 panel and SAME fixed1280 inference size. Record
   different training budgets and don't attribute extra-training gains purely
   to architecture. No automatic full-val or inference-scale sweep.
4. Freeze weights/config/geometry assets and selection BEFORE test inference.
   Use only supplied test RGB-D + metadata. No test annotations, Kaggle score
   tuning, or GT-based diagnostic re-ranking. Existing constant-camera-K
   assumption must be retained/disclosed unless official test K is supplied.
5. Export new unique directory; never overwrite old CSVs. Strict load, exact2112
   metadata IDs, columns image_id/PredictionString, category_id1, score[0,1],
   compressed COCO RLE1024x1024, positive-area masks only, <=100/image, [] for
   no detections; validate every record/RLE area, sample decode and SHA256.
6. Bring the CSV to local outputs and provide it. Generate, don't automatically
   upload or change browser permissions. User has not supplied a confirmed
   successful earlier Kaggle score/acceptance in this thread.

Previous prepared corrected CSV: d01 geometry model30k,1280 input,
full-val APs .5079599315. SHA256
44c8d67df29d31eedd548f21d108c53bcb1df8a03bfba5c61be657adea3bb343.
It is the strongest full-val-verified prior submission recipe, not proof of
global optimality or that newer P2/short-screen models are inferior.
Test was inference only. test_images.json supplied IDs/paths, not mask labels.
K was uniquely constant across supplied train/val; test K was assumed equal.
