# Screen V2: P2-conditioned queries, optional explicit depth geometry

The user explicitly requested autonomous result analysis, development and next
experiments without manual reminders. An active thread heartbeat checks every15
minutes (automation id rgb-d). Existing jobs must not be duplicated or interrupted;
only meaningful new results/failures/decisions should generate notifications.

## Candidates

- j01: original learned100 queries + bounded image-conditioned residual from
  stride4 mask/P2 features. Those features ALREADY contain existing RGB-D fusion;
  this is not an RGB-only ablation.
- j02: same design, adding aligned seven-channel geometry to the query pooling
  context. Reuses the parent's per-image whole-depth-branch dropout mask, no
  additional RNG draw and no bypass of depth dropout through the new input.
- Each learned query scores stride4 locations in32-channel latent space with a
  soft spatial prior anchored on a10x10 grid. Differentiable softmax pooling,
  not hard top-k proposals. Fixed grid promotes coverage; no GT center/pose/model
  metadata or validation labels enter model inputs or query selection.
- Output projection is zero initialized and residual bounded by.1*tanh(x).
  Initial masks/logits/auxiliary outputs must exactly match the trained model.
  Preserve original query features AND positions, original attention, all losses
  and matching. No additional supervision or decoder query count change.
- This tests image-conditioned query content, NOT a full MaskDINO reproduction
  or a guaranteed cure for1-pixel fragments. P2 already existed in baseline.

## Matched quick-screen protocol

Exactly the V1 frozen200-image panel, same d01 task checkpoint,+3000 updates,
batch2 FP32,1024,seed42, same data order/augmentations, AdamW and shared LR schedule.
New small query modules use LR1e-4 (10x base1e-5); trained model shared parameter
groups keep identical rates, backbone1e-6, geometry1e-5. New modules initialized
inside fork_rng so no change to subsequent training RNG stream.

Reuse completed V1 i00 control only after verifying all shared tensors, CPU/GPU
initial parity, unchanged upstream decoder/loss logic, training config, data
panel, trained weights and geometry asset hashes. Save control status hash and
comparison policy. This avoids wasting a third identical control run.

Measure at0/1000/2000/3000. Evaluate against matched i00(33k) APs.4645607379,
not original d01(30k) or a different resolution/full-val set. Retain diagnostics
for tiny/sparse cases and inspect stage-to-stage consistency. The held-out
confirmation panel remains unused during direction screening; no full val/test.

## Autonomous continuation boundary

When this suite ends, record exact results and commit a result note, then review
the evidence and implement the next justified quick experiment without waiting
for a user message. Promising directions get disjoint-scene small confirmation;
negative candidates are not promoted by choosing a lucky intermediate score.
Prioritize local geometry confidence/boundaries if query conditioning fails.
Never overwrite snapshots or outputs, delete best models, push remotes, download
external datasets, restart the container, or stop unrelated jobs automatically.

## Preflight evidence

-59 tests passed, including7 query-specific checks: zero-initialized/off decoder
  and all auxiliary outputs exactly equal official, original initialization RNG
  unchanged, shared model tensors preserved, staged inner/geometry gradients,
  batch isolation, shared dropout draw, optimizer LR isolation and strict export
  roundtrip preserving module-version metadata.
- GPU audit on2 real1024 views x2 candidates: exact masks/scores/classes/boxes
  against current control at zero residual.
- Parameters: control46,045,846; j01 46,072,854 (+27,008); j02 46,073,142
  (+27,296). Both below0.06% additional parameters; latency must still be measured.
- First smoke stopped BEFORE training on checkpoint export: converting state to
  a plain dict dropped PyTorch `_metadata`, causing the official head's historical
  v1->v2 conversion to rewrite valid v2 keys. Fixed with cpu_state_dict preserving
  OrderedDict metadata; no upstream load code or numerical weights changed.
  Failed smoke records retained. Successful recheck uses screen_v2_smoke_r2_*.
- Real recheck10-step image/geometry inner gradients are finite and nonzero.
  Main budget starts again from d01, not from the smoke checkpoints.
