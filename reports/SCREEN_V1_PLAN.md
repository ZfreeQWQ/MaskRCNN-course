# Screen V1: matched early-attention relaxation

User requested stopping the two V4 full-val scale scans and starting short model
iterations. V4 partial logs and all checkpoints are preserved. Input preprocessing
parity work from V4 remains valid; incomplete scale scans yield no AP claim.

## Exactly two arms, not another 30k run

- i00: original attention, continue trained V2 d01 for 3,000 updates.
- i01: same checkpoint, first two cross-attention masks expand allowed regions by
  3x3 max pooling on their respective attention grids; later layers remain official.
- R50, 1024 input, 100 queries, batch 2, FP32, seed 42, unchanged RGB-D/geometry
  features, synchronized augmentations, losses, matching, and optimizer grouping.
- Both fresh AdamW optimizers: LR 1e-5, backbone multiplier .1, warmup100,
  decays at added updates2000/2600. Existing task training30k + added3k, not new
  COCO-only initialization and not a claim of convergence after3k.
- No parameters added. No upstream Mask2Former source is edited. Opt-in decoder
  subclass retains state-dict keys and delegates all predictions/losses to upstream.
- Early dilation is applied at both training and inference. A grid cell is not an
  input pixel: dilation at the coarsest level spans a fairly broad RGB region.

## Frozen screening protocol

Use validation annotations ONLY to stratify scene sampling by tiny/sparse presence,
not to create model inputs or training examples. Pick100 scenes proportionally
across strata and2 views per scene =200 images. Freeze selection before scoring.
Create another200-image confirmation panel from disjoint scenes; do not evaluate
or tune against that panel in this queue. No train/val merge, no test access.

Evaluate both at added step0/1000/2000/3000 on exactly the same screen panel.
Step0 separates architecture-only effects from learning. Final comparison is
i01(33k) minus i00(33k), not i01(33k) minus frozen d01(30k).
Report APs, small AP50/AP75/AR, and independent best-IoU diagnostics for tiny,
sparse and elongated GT groups. The latter are oracle diagnostics, NOT COCO recall.
Small-screen results are direction screening, not full-val scores or significant
gains. Inspect trends and error groups before rejection or extension.

The bounded controller ends after both runs. It never launches full val, inference
scale sweeps, longer training, query experiments, or test submission automatically.
Record commit, immutable source archive, weight/panel/geometry asset hashes,
strict task-weight transfer and gradient audit for reproducibility.

Next priorities after analysis: P2+depth-guided query initialization; then local
geometry/reliability gating. P2 already participates in the current pixel decoder.

## Preflight evidence, 2026-10-25

- All52 tests passed, including5 new attention/panel/protocol checks and47
  existing data, geometry, matching, refinement and preprocessing regressions.
- Both GPU arms passed5 real training updates, strict633-tensor task checkpoint
  transfer with zero mismatches, fresh optimizer, finite nonzero query gradients,
  plus2-image pre/post inference. These smoke scores are NOT optimization results.
- Frozen screen:200 views/100 scenes,921 small GT,68 area<=64 GT,314 sparse
  small GT. Separate unused confirmation:200 views/100 disjoint scenes,722 small
  GT,56 tiny GT,249 sparse small GT. Panels generated before model scoring.
- V4 stopped by user at last logged1350/2584 (1536) and1500/2584 (detail1280).
  Incomplete scans have no AP result; checkpoints, source snapshots and logs kept.
