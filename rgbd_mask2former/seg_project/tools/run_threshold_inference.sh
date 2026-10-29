#!/usr/bin/env bash
# One unfiltered test inference, then CPU-only exports for all score cutoffs.
set -euo pipefail
cd /private/rgzndzy
out=seg_project/experiments/mask2former/threshold_sweep_20261025/test
mkdir -p "$out"
if [ -e "$out/master_test.json" ] || [ -e "$out/shard0.log" ]; then
    echo 'Existing inference artifacts found; inspect before retrying.' >&2
    exit 1
fi
pids=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" ./seg_project/run.sh python -u seg_project/tools/infer_m2f.py \
      --config seg_project/experiments/mask2former/d_screen/config.yaml \
      --weights seg_project/experiments/mask2former/d_screen/best_segm_APs.pth \
      --metadata seg_project/datasets/blender/annotations/test_images_depth.json \
      --split test --score-threshold 0 --num-shards 4 --shard-index "$gpu" \
      --output "$out/shard${gpu}.json" > "$out/shard${gpu}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    printf '%s\n' "$pid" > "$out/shard${gpu}.pid"
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
    echo 'At least one inference shard failed; check shard logs.' >&2
    exit 1
fi
./seg_project/run.sh python seg_project/tools/merge_predictions.py \
  --metadata seg_project/datasets/blender/annotations/test_images_depth.json \
  --split test --shards "$out/shard0.json" "$out/shard1.json" "$out/shard2.json" "$out/shard3.json" \
  --output "$out/master_test.json"
echo 'Master test predictions and complete manifest saved.'
