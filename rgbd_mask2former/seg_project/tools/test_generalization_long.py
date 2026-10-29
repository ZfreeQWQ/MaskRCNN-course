#!/usr/bin/env python
"""Test plateau decisions and the real checkpoint's continuation topology."""
import argparse
import json
from pathlib import Path
import unittest
from generalization_convergence import assess


class PolicyTests(unittest.TestCase):
    def test_two_epochs_cannot_trigger_stop(self):
        curve = [{"epoch": i, "APs": 55., "AP": 82., "lr": 1e-7} for i in (1, 2)]
        self.assertTrue(assess(curve)["continue_training"])

    def test_late_improvement_extends_training(self):
        curve = [{"epoch": i, "APs": 55. + (1. if i >= 11 else 0.), "AP": 82., "lr": 1e-7}
                 for i in range(1, 13)]
        self.assertTrue(assess(curve)["continue_training"])

    def test_three_flat_vals_with_reduced_lr_indicate_plateau(self):
        curve = [{"epoch": i, "APs": 55. + (0.05 if i >= 10 else 0.), "AP": 82., "lr": 1e-7}
                 for i in range(1, 13)]
        self.assertTrue(assess(curve)["plateau"])

    def test_high_lr_flatness_does_not_trigger_stop(self):
        curve = [{"epoch": i, "APs": 55., "AP": 82., "lr": 1e-5} for i in range(1, 13)]
        self.assertFalse(assess(curve)["plateau"])

    def test_late_regression_uses_best_not_only_last_score(self):
        curve = [{"epoch": i, "APs": 55. + (.5 if i == 11 else 0.), "AP": 82., "lr": 1e-7}
                 for i in range(1, 13)]
        self.assertTrue(assess(curve)["continue_training"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.policy_only:
        return
    import train_m2f
    import torch
    import rgbd_seg.m2f.frequency_fusion
    from detectron2.modeling import build_model
    from rgbd_seg.m2f.generalization import setup_cfg, GeneralizationTrainer, LocalMPCriterion
    from rgbd_seg.m2f.generalization_continuation import GeneralizationContinuationTrainer as Trainer
    from rgbd_seg.m2f.algorithm_continuation import audit_optimizer
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/mask2former/generalization_20261027"
    assert Trainer.build_train_loader.__func__ is GeneralizationTrainer.build_train_loader.__func__
    assert Trainer.build_optimizer.__func__ is GeneralizationTrainer.build_optimizer.__func__
    cfg = setup_cfg(out / "configs/crop_local.yaml", ("MODEL.DEVICE", "cpu"))
    torch.set_num_threads(4)
    checkpoint = torch.load(root / "experiments/mask2former/round4_control_20261027/model_final.pth",
                            map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    assert isinstance(model.criterion, LocalMPCriterion)
    optimizer = Trainer.build_optimizer(cfg, model)
    expected = checkpoint["trainer"]["_trainer"]["optimizer"]
    optimizer.load_state_dict(expected)
    audit = audit_optimizer(expected, optimizer.state_dict())
    # Every restored moment must match the shape of its actual model parameter.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                assert state["exp_avg"].shape == parameter.shape
                assert state["exp_avg_sq"].shape == parameter.shape
    report = {"valid": True, "convergence_policy_tests": result.testsRun,
              "continuation_mro": [cls.__name__ for cls in Trainer.__mro__],
              "focus_mapper_and_mature_optimizer_preserved": True, "real_checkpoint_strict_load": True,
              "real_checkpoint_completed_iterations": checkpoint["iteration"] + 1,
              "optimizer": audit, "all_optimizer_moment_shapes_verified": True}
    if not args.report:
        parser.error("--report required for real-checkpoint tests")
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
