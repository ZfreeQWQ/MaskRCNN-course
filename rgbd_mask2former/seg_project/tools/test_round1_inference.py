"""Independent synthetic checks for candidate ranking and resize/quality semantics."""
import unittest
import torch
import torch.nn.functional as F
from detectron2.modeling.postprocessing import sem_seg_postprocess
from rgbd_seg.m2f.experimental_inference import masks_and_quality,selection_indices,restore_logits


class RoundOneTests(unittest.TestCase):
    def test_quality_can_rescue_a_lower_class_score(self):
        classes=torch.tensor([.99,.8,.1])
        logits=torch.tensor([.001,10.,-10.])[:,None,None].expand(-1,4,4)
        masks,quality=masks_and_quality(logits)
        self.assertEqual(selection_indices(classes,quality,1,'class-first').tolist(),[0])
        self.assertEqual(selection_indices(classes,quality,1,'mask-first').tolist(),[1])
        self.assertEqual(quality[2].item(),0)
        self.assertFalse(masks[2].any())

    def test_resize_matches_padded_reference(self):
        torch.manual_seed(1)
        logits=torch.randn(7,9,12)
        reference=F.interpolate(logits[None],size=(40,52),mode='bilinear',align_corners=False)[0]
        reference=sem_seg_postprocess(reference,(37,49),61,73)
        actual=torch.cat([restore_logits(logits[i:i+2],(40,52),(37,49),(61,73)) for i in range(0,7,2)])
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)

    def test_top100_limit_and_score_order(self):
        classes=torch.linspace(.1,.9,200)
        quality=torch.linspace(.99,.01,200)
        ids=selection_indices(classes,quality,100,'mask-first')
        expected=torch.argsort(classes*quality,descending=True)[:100]
        self.assertEqual(set(ids.tolist()),set(expected.tolist()))
        self.assertEqual(len(ids),100)


if __name__=='__main__': unittest.main()
