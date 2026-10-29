"""Verify all eight independent switch combinations without building a model."""
import argparse
import copy
import itertools
from pathlib import Path
import sys
import unittest

from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizations.configure import apply_optimizations


class SwitchTests(unittest.TestCase):
    def base(self):
        return Config(dict(model=dict(panoptic_head=dict(type='Mask2FormerHead')),
                           train_dataloader=dict(dataset=dict(pipeline=[dict(type='RandomCrop', crop_size=(512, 512))])),
                           val_dataloader=dict(dataset=dict(pipeline=[dict(type='Resize', scale=(853, 512))]))))

    def test_every_combination_is_independent(self):
        for crop, small, boundary in itertools.product((False, True), repeat=3):
            with self.subTest(crop=crop, small=small, boundary=boundary):
                cfg = self.base()
                original = copy.deepcopy(cfg.to_dict())
                args = argparse.Namespace(small_object_crop=crop, small_object_loss=small,
                                          boundary_loss=boundary, small_crop_prob=0.5,
                                          small_area_threshold=1024.0, small_loss_max_weight=2.0,
                                          boundary_loss_weight=0.1)
                apply_optimizations(cfg, args)
                self.assertEqual(cfg.val_dataloader.to_dict(), original['val_dataloader'])
                self.assertEqual(cfg.train_dataloader.dataset.pipeline[0]['type'],
                                 'SmallObjectAwareCrop' if crop else 'RandomCrop')
                head = cfg.model.panoptic_head
                if small or boundary:
                    self.assertEqual(head.type, 'WeightedBoundaryMask2FormerHead')
                    self.assertEqual(head.small_object_loss.enabled, small)
                    self.assertEqual(head.boundary_loss.enabled, boundary)
                else:
                    self.assertEqual(head.to_dict(), original['model']['panoptic_head'])
                if not any((crop, small, boundary)):
                    self.assertEqual(cfg.to_dict(), original)


if __name__ == '__main__':
    unittest.main()
