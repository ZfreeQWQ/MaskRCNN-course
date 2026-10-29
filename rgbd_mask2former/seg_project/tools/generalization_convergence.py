"""Pure convergence policy: observed val plateau, never a proof of convergence."""
import math


def assess(curve, minimum_epoch=12, delta=.2, patience=3):
    """curve rows: epoch, APs, AP, end-of-epoch lr; all are full val only."""
    rows = sorted(curve, key=lambda row: row["epoch"])
    if not rows or any(not math.isfinite(row["APs"]) for row in rows):
        raise ValueError("A finite full-validation curve is required")
    last = rows[-1]
    if last["epoch"] < minimum_epoch or len(rows) < patience + 1:
        return {"plateau": False, "continue_training": True, "reason": "minimum_budget_not_reached"}
    before = [r for r in rows if r["epoch"] <= last["epoch"] - patience]
    recent = [r for r in rows if r["epoch"] > last["epoch"] - patience]
    if not before or len(recent) < patience:
        return {"plateau": False, "continue_training": True, "reason": "insufficient_full_val_history"}
    improvement = max(r["APs"] for r in recent) - max(r["APs"] for r in before)
    # The current cycle must have actually reached its reduced-LR stage.
    low_lr = last.get("lr", math.inf) <= 1.01e-6
    plateau = improvement < delta and low_lr
    return {"plateau": plateau, "continue_training": not plateau,
            "reason": "three_full_vals_without_0.2_point_gain_at_reduced_lr" if plateau else "recent_gain_or_lr_not_reduced",
            "recent_best_gain_APs": improvement, "patience_full_evaluations": patience,
            "minimum_gain_points": delta, "last_epoch": last["epoch"], "reduced_lr_observed": low_lr,
            "interpretation": "empirical validation plateau; not proof of global convergence"}
