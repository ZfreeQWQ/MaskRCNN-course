"""Predeclared one-winner confirmation; never search the confirmation panel."""


def select_winner(jobs, control_scores):
    by_step = {v["additional_training_step"]: v["mask_aps"] for v in control_scores}
    assert 3000 in by_step
    eligible = []
    for name, job in jobs.items():
        if job["status"] != "completed":
            raise ValueError("Wait for every candidate's fixed final budget")
        scores = job["screen_metrics"]
        actual = {v["additional_training_step"]: v["mask_aps"] for v in scores}
        assert 3000 in actual
        delta = actual[3000] - by_step[3000]
        positive_steps = sum(actual[k] > by_step[k] for k in (1000, 2000, 3000))
        if delta > 0 and positive_steps >= 2:
            eligible.append((delta, name))
    return max(eligible)[1] if eligible else None


def validate_panels(screen, confirmation):
    if screen["split"] != "val" or confirmation["split"] != "val":
        raise ValueError("Only validation panels are allowed")
    if set(screen["scenes"]) & set(confirmation["scenes"]):
        raise ValueError("Confirmation scenes overlap direction-screening scenes")
    for panel in (screen, confirmation):
        ids = panel["image_ids"]
        if len(ids) != 200 or len(ids) != len(set(ids)) or len(panel["scenes"]) != 100:
            raise ValueError("Protocol requires200 distinct images /100 scenes")
    if set(screen["image_ids"]) & set(confirmation["image_ids"]):
        raise ValueError("Confirmation images overlap direction-screening images")
