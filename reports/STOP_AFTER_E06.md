# User-requested training stop — 2026-10-23

Finish e06_depth_p2_gate (30k updates plus final full-val evaluation), export
the model and record results, then stop. Do NOT start e07_resolution1280 or
any new training before the user authorizes it. The next phase is code and
error analysis, especially multi-scale RGB-D fusion inspired by SA-Gate/CMX.

The original seven-job dispatcher was suspended and replaced by a finish-only
observer. It adopts the already-running e06 trainer, rather than stopping or
resuming training. Immutable training source remains commit `28b288d`.

The observer is incapable of starting a trainer. A persistent
`runs/ablations_v1/queue_control.json` also guards the normal dispatcher
against launching any formal experiment after e06.

After completion, suite status should be `stopped_by_user`, e06 `completed`,
and e07 `deferred_by_user` (only its earlier preflight smoke test is retained).
