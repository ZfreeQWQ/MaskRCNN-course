import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from course_demo import check_inputs, checked_image_path, encode_predictions


class CourseDemoTests(unittest.TestCase):
    def test_uint16_pair_and_scale_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            rgb,depth = Path(directory)/'rgb.png',Path(directory)/'depth.png'
            Image.fromarray(np.zeros((8,10,3),dtype=np.uint8)).save(rgb)
            Image.fromarray(np.ones((8,10),dtype=np.uint16)*12000).save(depth)
            result = check_inputs(rgb,depth,.0001,0)
            self.assertEqual(result['original_shape'],[8,10])
            self.assertEqual(len(result['depth_sha256']),64)
            for scale in (0,-1,float('nan'),float('inf')):
                with self.assertRaises(ValueError):check_inputs(rgb,depth,scale,0)
            Image.fromarray(np.zeros((8,10),dtype=np.uint8)).save(depth)
            with self.assertRaises(ValueError):check_inputs(rgb,depth,.0001,0)

    def test_mismatched_geometry_and_test_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rgb,depth=root/'rgb.png',root/'depth.png'
            Image.fromarray(np.zeros((8,10,3),dtype=np.uint8)).save(rgb)
            Image.fromarray(np.ones((7,10),dtype=np.uint16)).save(depth)
            with self.assertRaises(ValueError):check_inputs(rgb,depth,.0001,0)
            (root/'test').mkdir()
            Image.fromarray(np.zeros((8,10,3),dtype=np.uint8)).save(root/'test/rgb.png')
            with self.assertRaises(ValueError):checked_image_path(root/'test/rgb.png')

    def test_rle_roundtrip_scores_boxes_empty_and_json(self):
        masks=np.zeros((3,8,10),dtype=bool)
        masks[0,2:4,3:6]=True;masks[2,5,7]=True
        rows=encode_predictions(masks,np.array([.1,.9,.8]),123)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['score'],.8)
        self.assertEqual(rows[0]['bbox'],[7.,5.,1.,1.])
        np.testing.assert_array_equal(mask_utils.decode(rows[1]['segmentation']).astype(bool),masks[0])
        self.assertEqual(rows,json.loads(json.dumps(rows)))
        self.assertEqual(encode_predictions(masks[:0],np.array([]),1),[])

    def test_invalid_prediction_shape_scores_and_limit(self):
        with self.assertRaises(ValueError):encode_predictions(np.zeros((101,2,2)),np.ones(101),1)
        with self.assertRaises(ValueError):encode_predictions(np.zeros((2,2)),np.ones(2),1)
        with self.assertRaises(ValueError):encode_predictions(np.zeros((1,2,2)),np.array([float('nan')]),1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
