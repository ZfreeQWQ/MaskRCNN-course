# Kaggle empty-instance format correction

User reported: empty instance masks are not allowed; use[] for no detections.
Audit of the original exported CSV found29413 zero-area masks among211200
instances, present in2104/2112 images, including score0 all-background masks.
This was an exporter/validator omission: validate_prediction allowed area0.
It is not evidence of a model-load failure and does not require new inference.
The original generation report's checks were insufficient for Kaggle's stricter
nonempty-instance contract. Original CSV is retained for audit, not resubmission.

Fix: export_submission.py skips all-background masks; validator rejects zero
area explicitly while accepting an empty image's JSON[]. repair_submission.py
filters only zero-area RLE instances from existing predictions. No score cutoff,
class, nonempty mask, ordering, model, training or inference setting is changed.
One-pixel masks and even score0 NONEMPTY masks remain intact.
Seven submission tests cover empty-instance rejection, [] acceptance, exact
survivor preservation, one-pixel masks, metadata/order/duplicate safeguards,
and an entirely empty image becoming[]. Existing outputs are not overwritten.

Corrected output: runs/submission_d01_1280_v2/submission.csv
2112 image rows,181787 retained instances,29413 removed empty masks.
Min/max instances per image7/100; no image in this actual file is completely
empty, but the [] path is regression-tested. Every retained RLE has positive
area; decode roundtrip checks every100th image.
65,988,640 bytes; SHA256:
44c8d67df29d31eedd548f21d108c53bcb1df8a03bfba5c61be657adea3bb343.
Original SHA256:
d1e1803e6bc4c58128ccd393262fa219b7506110b3394778820cc7b95ddb64a3.

No successful Kaggle acceptance or score is claimed by this repair report.
The browser extension's missing file-URL access remains a separate upload
blocker unless the user has since enabled it; do not change that permission.
