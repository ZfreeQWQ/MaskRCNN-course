"""Synthetic CPU tests; no data cache, weights, GPU, training or registration.

Run in the existing remote Mask2Former environment:
    python -m unittest -v test_roi_v2
"""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from detectron2.projects.point_rend.point_features import point_sample
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import (
    HungarianMatcher,
    batch_dice_loss_jit,
    batch_sigmoid_ce_loss_jit,
)

from course.roi_v2 import ROICriterion, ROIHungarianMatcher, sample_roi_points


def make_target():
    masks = torch.zeros(3, 24, 24)
    masks[0, 3:7, 3:7] = 1
    masks[1, 6:21, 5:20] = 1
    masks[2, 16:20, 17:21] = 1
    return dict(
        masks=masks, labels=torch.zeros(3, dtype=torch.int64),
        small=torch.tensor([True, False, True]),
        original_area=torch.tensor([16., 4096., 64.]),
        # Already-expanded boxes on a common padded canvas.
        roi_boxes=torch.tensor([[.10, .10, .32, .32], [.15, .20, .90, .95],
                                [.65, .60, .95, .90]]),
    )


def empty_target():
    return dict(masks=torch.empty(0, 24, 24), labels=torch.empty(0, dtype=torch.int64),
                small=torch.empty(0, dtype=torch.bool), original_area=torch.empty(0),
                roi_boxes=torch.empty(0, 4))


def make_outputs(batch=1, queries=4, grad=False):
    return dict(pred_logits=torch.randn(batch, queries, 2, requires_grad=grad),
                pred_masks=torch.randn(batch, queries, 12, 12, requires_grad=grad))


def make_criterion(kind=ROICriterion, **kwargs):
    return kind(1, ROIHungarianMatcher(2., 5., 5., 64, roi_points=16),
                {"loss_ce": 2., "loss_mask": 5., "loss_dice": 5.}, .1,
                ["labels", "masks"], 40, 3., .75, **kwargs)


class ROIPointTests(unittest.TestCase):
    def test_shape_bounds_and_no_second_expansion(self):
        boxes = torch.tensor([[.2, .3, .4, .6], [.7, .1, .9, .2]])
        generator = torch.Generator().manual_seed(13)
        expected_random = torch.rand(2, 256, 2, generator=generator)
        actual = sample_roi_points(boxes, 256, generator=torch.Generator().manual_seed(13))
        expected = boxes[:, None, :2] + expected_random * (boxes[:, None, 2:] - boxes[:, None, :2])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (2, 256, 2))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual >= boxes[:, None, :2]).all())
        self.assertTrue((actual <= boxes[:, None, 2:]).all())
        self.assertEqual(tuple(sample_roi_points(torch.empty(0, 4), 256).shape), (0, 256, 2))
        self.assertEqual(tuple(sample_roi_points(boxes, 0).shape), (2, 0, 2))

    def test_uniform_box_sampling_has_foreground_and_background(self):
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 12:20, 12:20] = 1
        points = sample_roi_points(torch.tensor([[.25, .25, .75, .75]]), 1024,
                                   generator=torch.Generator().manual_seed(2))
        labels = point_sample(mask, points, align_corners=False)
        self.assertTrue((labels == 1).any())
        self.assertTrue((labels == 0).any())

    def test_invalid_shape_finite_and_geometry(self):
        for boxes in (torch.zeros(4), torch.zeros(1, 3), torch.ones(1, 4, dtype=torch.int64),
                      torch.tensor([[0., 0., float("nan"), 1.]]),
                      torch.tensor([[-.1, 0., 1., 1.]]), torch.tensor([[0., 0., 1.1, 1.]]),
                      torch.tensor([[.5, 0., .5, 1.]]), torch.tensor([[.8, 0., .2, 1.]])):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                sample_roi_points(boxes, 8)
        with self.assertRaises(ValueError):
            sample_roi_points(torch.tensor([[0., 0., 1., 1.]]), -1)


class ROIMatcherTests(unittest.TestCase):
    def test_defaults_and_exact_zero_weight_official_equivalence(self):
        matcher = ROIHungarianMatcher(2., 5., 5., 64)
        self.assertEqual((matcher.roi_points, matcher.roi_weight), (256, .3))
        outputs, targets = make_outputs(batch=2), [make_target(), make_target()]
        official = HungarianMatcher(2., 5., 5., 64)
        disabled = ROIHungarianMatcher(2., 5., 5., 64, roi_weight=0.)
        # Disabled mode accepts official targets without any ROI metadata.
        targets = [{k: t[k] for k in ("labels", "masks")} for t in targets]
        torch.manual_seed(71)
        expected = official(outputs, targets)
        expected_rng = torch.rand(8)
        torch.manual_seed(71)
        actual = disabled(outputs, targets)
        actual_rng = torch.rand(8)
        for pair_a, pair_b in zip(actual, expected):
            for a, b in zip(pair_a, pair_b):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    def test_zero_weight_mixed_empty_batch_preserves_official_rng(self):
        outputs = make_outputs(batch=2)
        targets = [empty_target(), make_target()]
        torch.manual_seed(41)
        expected = HungarianMatcher(2., 5., 5., 64)(outputs, targets)
        expected_rng = torch.rand(8)
        torch.manual_seed(41)
        actual = ROIHungarianMatcher(2., 5., 5., 64, roi_weight=0.)(outputs, targets)
        for actual_pair, expected_pair in zip(actual, expected):
            for a, b in zip(actual_pair, expected_pair):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(8), expected_rng, rtol=0, atol=0)

    def test_separate_normalized_costs_small_columns_and_unchanged_class_cost(self):
        outputs, target = make_outputs(), make_target()
        matcher = ROIHungarianMatcher(2., 5., 7., 64, roi_points=8, roi_weight=.3)
        logits, masks = outputs["pred_logits"][0], outputs["pred_masks"][0]
        torch.manual_seed(19)
        coords = torch.rand(1, 64, 2)
        gt = point_sample(target["masks"][:, None], coords.repeat(3, 1, 1), align_corners=False)[:, 0]
        pred = point_sample(masks[:, None], coords.repeat(4, 1, 1), align_corners=False)[:, 0]
        global_bce = batch_sigmoid_ce_loss_jit(pred, gt)
        global_dice = batch_dice_loss_jit(pred, gt)
        class_cost = -logits.softmax(-1)[:, target["labels"]]
        roi_bce, roi_dice = torch.full((4, 2), .25), torch.full((4, 2), .75)
        expected_bce, expected_dice = global_bce.clone(), global_dice.clone()
        expected_bce[:, [0, 2]] = .7 * global_bce[:, [0, 2]] + .3 * roi_bce
        expected_dice[:, [0, 2]] = .7 * global_dice[:, [0, 2]] + .3 * roi_dice
        torch.manual_seed(19)
        with patch.object(matcher, "_roi_costs", return_value=(roi_bce, roi_dice)):
            actual = matcher._cost_matrix(logits, masks, target)
        expected = 5 * expected_bce + 2 * class_cost + 7 * expected_dice
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        original = 5 * global_bce + 2 * class_cost + 7 * global_dice
        torch.testing.assert_close(actual[:, 1], original[:, 1], rtol=0, atol=0)

    def test_per_gt_costs_match_independent_reference_and_chunking(self):
        target = make_target()
        masks = torch.randn(4, 1, 12, 12)
        ids = torch.tensor([0, 2])
        boxes = target["roi_boxes"]
        torch.manual_seed(23)
        all_coords = sample_roi_points(boxes[ids], 32)
        reference_bce, reference_dice = [], []
        for j, gt_id in enumerate(ids):
            coords = all_coords[j:j + 1]
            pred = point_sample(masks, coords.expand(4, -1, -1), align_corners=False)[:, 0]
            gt = point_sample(target["masks"][gt_id:gt_id + 1, None], coords, align_corners=False)[0, 0]
            reference_bce.append(F.binary_cross_entropy_with_logits(pred, gt.expand_as(pred), reduction="none").mean(-1))
            prob = pred.sigmoid()
            reference_dice.append(1 - (2 * (prob * gt).sum(-1) + 1) / (prob.sum(-1) + gt.sum() + 1))
        for chunk_size in (1, 2, 16):
            matcher = ROIHungarianMatcher(num_points=64, roi_points=32, roi_chunk_size=chunk_size)
            torch.manual_seed(23)
            with patch("course.roi_v2.point_sample", wraps=point_sample) as sampler:
                bce, dice = matcher._roi_costs(masks, target["masks"][:, None], boxes, ids)
            torch.testing.assert_close(bce, torch.stack(reference_bce, dim=1))
            torch.testing.assert_close(dice, torch.stack(reference_dice, dim=1))
            for call in sampler.call_args_list:
                full_masks = call.args[0]
                self.assertEqual(full_masks.ndim, 4)
                # Query calls always use Q masks; GT calls use <=chunk_size.
                self.assertLessEqual(len(full_masks), max(4, chunk_size))

    def test_another_gt_box_cannot_change_first_gt_column(self):
        target = make_target()
        other = {key: value.clone() for key, value in target.items()}
        other["roi_boxes"][2] = torch.tensor([.01, .01, .20, .20])
        other["masks"][2] = 1 - other["masks"][2]
        outputs = make_outputs()
        matcher = ROIHungarianMatcher(num_points=64, roi_points=32, roi_chunk_size=1)
        torch.manual_seed(31)
        a = matcher._cost_matrix(outputs["pred_logits"][0], outputs["pred_masks"][0], target)
        torch.manual_seed(31)
        b = matcher._cost_matrix(outputs["pred_logits"][0], outputs["pred_masks"][0], other)
        torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, 2], b[:, 2]))

    def test_empty_targets_zero_queries_and_mixed_batch(self):
        matcher = ROIHungarianMatcher(num_points=64)
        for outputs, targets in ((make_outputs(), [empty_target()]),
                                 (make_outputs(queries=0), [make_target()]),
                                 (make_outputs(batch=2), [empty_target(), make_target()])):
            indices = matcher(outputs, targets)
            for (src, dst), target in zip(indices, targets):
                self.assertEqual(len(src), min(outputs["pred_masks"].shape[1], len(target["labels"])))
                self.assertEqual(src.dtype, torch.int64)
                self.assertEqual(dst.dtype, torch.int64)
                self.assertEqual(src.device.type, "cpu")
                self.assertEqual(len(src), len(dst))

    def test_shape_and_nonfinite_rejections(self):
        matcher, target = ROIHungarianMatcher(num_points=64), make_target()
        bad = make_outputs()
        bad["pred_masks"] = torch.zeros(1, 5, 12, 12)
        with self.assertRaises(ValueError):
            matcher(bad, [target])
        for key in ("pred_logits", "pred_masks"):
            bad = make_outputs()
            bad[key].fill_(float("nan"))
            with self.subTest(key=key), self.assertRaises(ValueError):
                matcher(bad, [target])
        target["small"] = torch.tensor([True])
        with self.assertRaises(ValueError):
            matcher(make_outputs(), [target])


class ROICriterionTests(unittest.TestCase):
    def test_zero_fraction_exact_official_loss_and_rng(self):
        outputs, targets = make_outputs(grad=True), [make_target()]
        indices = [(torch.tensor([0, 2, 3]), torch.tensor([2, 0, 1]))]
        original = make_criterion(SetCriterion)
        disabled = make_criterion(roi_fraction=0.)
        torch.manual_seed(42)
        expected = original.loss_masks(outputs, targets, indices, 3)
        expected_rng = torch.rand(8)
        torch.manual_seed(42)
        actual = disabled.loss_masks(outputs, targets, indices, 3)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(8), expected_rng, rtol=0, atol=0)

    def test_only_matched_small_points_are_replaced_in_own_boxes(self):
        criterion, target = make_criterion(), make_target()
        indices = [(torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]))]
        base = torch.rand(3, criterion.num_points, 2)
        with patch("course.roi_v2.get_uncertain_point_coords_with_randomness", return_value=base.clone()) as uncertain:
            with patch("course.roi_v2.point_sample", wraps=point_sample) as sampler:
                criterion.loss_masks(make_outputs(), [target], indices, 3)
        self.assertEqual(uncertain.call_args.args[2], criterion.num_points)
        actual = sampler.call_args_list[0].args[1]
        k = int(.3 * criterion.num_points)
        slots = torch.div(torch.arange(k) * criterion.num_points, k, rounding_mode="floor")
        retained = torch.ones(criterion.num_points, dtype=torch.bool)
        retained[slots] = False
        # Matched row 0 is LARGE despite its source query being 2.
        torch.testing.assert_close(actual[0], base[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[:, retained], base[:, retained], rtol=0, atol=0)
        for row, gt in ((1, 2), (2, 0)):
            points, box = actual[row, slots], target["roi_boxes"][gt]
            self.assertTrue((points >= box[:2]).all() and (points <= box[2:]).all())
        self.assertTrue(torch.equal(actual, sampler.call_args_list[1].args[1]))

    def test_all_large_matches_keep_official_samples_and_losses(self):
        outputs, target = make_outputs(), make_target()
        target["small"].fill_(False)
        indices = [(torch.tensor([0, 1, 2]), torch.arange(3))]
        torch.manual_seed(12)
        expected = make_criterion(SetCriterion).loss_masks(outputs, [target], indices, 3)
        torch.manual_seed(12)
        actual = make_criterion().loss_masks(outputs, [target], indices, 3)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_area_weight_default_off_and_all_target_normalization(self):
        outputs = dict(pred_masks=torch.zeros(1, 2, 12, 12, requires_grad=True))
        target = make_target()
        target["original_area"] = torch.tensor([16., 1024., 64.])  # weights [2, 1, 2]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]  # third GT unmatched
        weighted = make_criterion(roi_fraction=0., area_weight=True)
        loss = weighted.loss_masks(outputs, [target], indices, 3)
        expected = torch.tensor(2.).log() * (2 + 1) / (3 * ((2 + 1 + 2) / 3))
        torch.testing.assert_close(loss["loss_mask"], expected)
        # Disabled area weighting must not even read original_area.
        del target["original_area"]
        unweighted = make_criterion(roi_fraction=0.)
        torch.testing.assert_close(unweighted.loss_masks(outputs, [target], indices, 3)["loss_mask"],
                                   torch.tensor(2.).log() * 2 / 3)

    def test_empty_targets_and_empty_matches_have_differentiable_zeros(self):
        criterion = make_criterion(area_weight=True)
        for target in (empty_target(), make_target()):
            outputs = make_outputs(grad=True)
            indices = [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))]
            loss = criterion.loss_masks(outputs, [target], indices, max(1, len(target["labels"])))
            self.assertTrue(all(value.item() == 0 for value in loss.values()))
            sum(loss.values()).backward()
            self.assertTrue(torch.isfinite(outputs["pred_masks"].grad).all())
            self.assertEqual(torch.count_nonzero(outputs["pred_masks"].grad).item(), 0)
        # Inherited full forward still produces finite no-object classification.
        outputs = make_outputs(grad=True)
        loss = criterion(outputs, [empty_target()])
        self.assertTrue(all(torch.isfinite(value) for value in loss.values()))

    def test_full_forward_auxiliary_losses_and_finite_cpu_gradients(self):
        outputs = make_outputs(batch=2, grad=True)
        auxiliary = make_outputs(batch=2, grad=True)
        outputs["aux_outputs"] = [auxiliary]
        loss = make_criterion(area_weight=True)(outputs, [empty_target(), make_target()])
        self.assertEqual(set(loss), {"loss_ce", "loss_mask", "loss_dice", "loss_ce_0", "loss_mask_0", "loss_dice_0"})
        self.assertTrue(all(torch.isfinite(value) for value in loss.values()))
        sum(loss.values()).backward()
        for layer in (outputs, auxiliary):
            for key in ("pred_masks", "pred_logits"):
                self.assertTrue(torch.isfinite(layer[key].grad).all())
                self.assertGreater(layer[key].grad.abs().sum().item(), 0)

    def test_single_process_shape_finite_and_area_guards(self):
        criterion, target, outputs = make_criterion(), make_target(), make_outputs()
        indices = [(torch.arange(3), torch.arange(3))]
        with patch("course.roi_v2.get_world_size", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "single-process"):
                criterion(outputs, [target])
            with self.assertRaisesRegex(RuntimeError, "single-process"):
                criterion.loss_masks(outputs, [target], indices, 3)
        target["roi_boxes"] = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            criterion.loss_masks(outputs, [target], indices, 3)
        target = make_target()
        target["original_area"][0] = 0
        with self.assertRaises(ValueError):
            make_criterion(area_weight=True).loss_masks(outputs, [target], indices, 3)
        outputs["pred_masks"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            criterion.loss_masks(outputs, [make_target()], indices, 3)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
