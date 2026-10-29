import unittest
import numpy as np
import torch
from pycocotools import mask as mu
import m2f_boundary_refiner as r


def row(mask):
    return dict(image_id=1, category_id=1, score=.75, segmentation=r.encode(mask))


class Contracts(unittest.TestCase):
    def test_identity_and_parameters(self):
        torch.manual_seed(r.SEED)
        model = r.Refiner().eval()
        x = torch.rand(2, 6, 128, 128)
        x[:, 5] = x[:, 5] > .5
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x) > 0, x[:, 5].bool()))
        self.assertTrue(115000 < sum(p.numel() for p in model.parameters()) < 125000)

    def test_real_backward_finite(self):
        model = r.Refiner()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        x = torch.rand(2, 6, 128, 128)
        x[:, 5] = x[:, 5] > .5
        target = 1-x[:, 5]
        change = torch.ones(2, 128, 128)
        # First step changes zero-initialized output; second reaches encoder.
        for _ in range(2):
            opt.zero_grad()
            loss = r.band_loss(model(x), target, change)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            opt.step()
        self.assertTrue(model.enc1[0].weight.grad.abs().sum() > 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_native_edge_crop(self):
        for x, y in ((0, 0), (1000, 1000), (511, 511)):
            m = np.zeros((1024, 1024), bool);m[y:y+10, x:x+10] = True
            xy = r.window(row(m))
            self.assertIsNotNone(xy)
            self.assertEqual(int(m[xy[1]:xy[1]+128, xy[0]:xy[0]+128].sum()), 100)

    def test_no_large_or_long_crop(self):
        m = np.zeros((1024, 1024), bool);m[10, 10:140] = True
        self.assertIsNone(r.window(row(m)))
        m[10:60, 10:60] = True
        self.assertIsNone(r.window(row(m)))

    def test_band_four_pixels(self):
        m = np.zeros((128, 128), bool);m[40:60, 40:60] = True
        b = r.band(m)
        self.assertTrue(b[36, 40]);self.assertFalse(b[35, 40])
        self.assertTrue(b[43, 50]);self.assertFalse(b[44, 50])
        self.assertFalse(b[64, 40])

    def test_composition_preserves_outside_and_fields(self):
        m = np.zeros((1024, 1024), bool);m[500:510, 500:510] = True
        a = row(m);xy = r.window(a);x, y = xy
        c = m[y:y+128, x:x+128];b = r.band(c)
        z, fallback = r.compose(a, xy, c, b, np.ones_like(c))
        out = mu.decode(z["segmentation"]).astype(bool)
        region = np.zeros_like(m);region[y:y+128, x:x+128] = b
        self.assertTrue(np.array_equal(out[~region], m[~region]))
        self.assertEqual(z["score"], a["score"]);self.assertFalse(fallback)
        self.assertTrue(np.array_equal(mu.decode(r.encode(out)), out))

    def test_empty_fallback(self):
        m = np.zeros((1024, 1024), bool);m[100, 100] = True
        a = row(m);x, y = r.window(a);c = m[y:y+128, x:x+128]
        out, flag = r.compose(a, (x,y), c, r.band(c), np.zeros_like(c))
        self.assertTrue(flag);self.assertEqual(out, a)

    def test_inputs_no_gt_and_no_resize(self):
        rgb = np.zeros((128, 128, 3), np.uint8);rgb[5,6] = [255,127,0]
        depth = np.zeros((128,128), np.float32);depth[5,6] = 2.
        valid = depth > 0;coarse = np.zeros((128,128), bool);coarse[5,6] = True
        x = r.patch_input(rgb, depth, valid, coarse)
        self.assertEqual(x.shape, (6,128,128));self.assertEqual(x[0,5,6], 1.)
        self.assertAlmostEqual(float(x[3,5,6]), np.log(3), places=6)
        self.assertEqual(x[4,5,6], 1.);self.assertEqual(x[5,5,6], 1.)

    def test_infer_identity(self):
        m = np.zeros((1024,1024), bool);m[123:133,234:244] = True
        rgbd = (np.zeros((1024,1024,3), np.uint8), np.zeros((1024,1024),np.float32),
                np.zeros((1024,1024),bool))
        out, stat = r.infer([row(m)], rgbd, r.Refiner().eval(), "cpu")
        self.assertEqual(out, [row(m)]);self.assertEqual(stat["eligible"], 1)


if __name__ == "__main__":
    unittest.main()
