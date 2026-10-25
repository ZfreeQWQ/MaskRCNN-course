# Screen V3: local relative geometry, with an explicit reliability ablation

Evidence: P2 query direction repeats modestly on an independent scene panel,
but tiny high-IoU remains1/56 and elongated components lose coverage. Existing
four-scale branch encodes absolute depth/height/normals; adding geometry to a
global query pool did less than P2 alone. Hypothesis: preserve local relative
edges at high resolution and avoid pseudo-normal boundaries near occlusion.

## Two isolated candidates

- k01_local_contrast: signed adjacent relative depth differences (+/-x/y),
  normal changes x/y, supplied relative edge, adjacent support-height changes.
  Reconstruct physical channels with existing TRAIN-ONLY mean/std; all valid
  pairs masked, no GT positions/IDs/metadata. Max-pool eight contrast channels
  to P2 so1px edges aren't averaged away; average valid/normal confidence.
- k02_reliable_contrast: identical network and10 derived channels, additionally
  exclude differences between missing/invalid normals and reduce contrasts
  without neighborhood depth support. Keep real high depth discontinuities;
  do not suppress every object boundary as unreliable.
- Small CNN10->32, depthwise3x3, zero-initialized32->256 projection adds to
  P2 mask features before ORIGINAL decoder heads. Same R50,100 queries,
  attention, matching and loss. No new query initialization in this isolated
  ablation. Initial full outputs must equal d01 exactly.
- Reuse the four-scale geometry branch's existing per-image depth-dropout
  draw for local geometry too. New modules constructed inside fork_rng.

## Fixed comparison

Both start again at d01(30k), not j01(33k) or smoke weights. +3000 updates each,
same200 screen views,1024,batch2FP32,seed42, shared baseLR1e-5/backbone1e-6,
same schedule and data/geometry hashes. New local moduleLR1e-4; all shared
optimizer rates/clipping/decay unchanged. Reuse audited i00 control(final APs
.4645607379). Evaluate0/1000/2000/3000 only on direction screen, no full val/test.

Compare all-small APs/AP75/AR and tiny/sparse/elongated oracle diagnostics.
No best-middle-checkpoint promotion. Do not tune on the already consumed j01
confirmation panel, don't evaluate it again for these candidates. This is an
engineering hypothesis, not a reproduction or guarantee from a named paper.

The runner supports an explicit suite/candidate/audit/smoke prefix while keeping
old defaults. Launch V3 only after all regressions, real GPU zero-output parity,
metadata-preserving strict checkpoint reload and10-step inner gradients pass.
Preserve all older snapshots and model files. Heartbeat continues autonomously.

## Reproducible entry points / preflight

- All68 tests passed (6 local-geometry tests plus62 prior regressions).
- Real GPU2 views x2 modes: exact masks/scores/classes/boxes at zero residual.
  Both candidates46,057,590 parameters vs46,045,846 control (+11,744).
- The one-pixel edge unit test explicitly checks asymmetric relative-depth
  denominators: .20->.22 uses.20 while .22->.20 uses.22. It does not assume
  equal positive/negative normalized magnitudes.
- GPU audit: `audit_query_gpu.py --candidates k01_local_contrast
  k02_reliable_contrast --output runs/screen_v3_gpu_audit.json`.
- Formal entry: `run_screen_v2.py --suite screen_v3 --candidates
  k01_local_contrast k02_reliable_contrast --gpu-audit
  runs/screen_v3_gpu_audit.json --smoke-prefix screen_v3_smoke_`.
- Each10-step real smoke must finish with finite nonzero new-module inner
  gradients and strict checkpoint reload before the entry is invoked.
