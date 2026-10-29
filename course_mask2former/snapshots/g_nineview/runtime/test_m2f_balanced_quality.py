import unittest
import numpy as np
import m2f_quality_selector as original
import m2f_balanced_quality as balanced


class BalancedTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(60, 13))
        self.y = np.tanh(self.x[:, 0])/2

    def test_unit_weights_reproduce_original(self):
        a = original.fit(self.x, self.y)
        b = balanced.fit_weighted(self.x, self.y, np.ones(60))
        np.testing.assert_allclose(original.predict(self.x, a), original.predict(self.x, b), atol=1e-12)

    def test_equal_group_totals(self):
        pairs = [{"gt_area": a} for a in [1, 256, 257, 1024, 1025, None, None]]
        group, counts, w = balanced.groups_and_weights(pairs)
        self.assertEqual(counts.tolist(), [2, 2, 3])
        np.testing.assert_allclose([w[group == i].sum() for i in range(3)], [7/3]*3)

    def test_missing_group_fails(self):
        with self.assertRaises(AssertionError):
            balanced.groups_and_weights([{"gt_area": 1}])

    def test_standardizer_unchanged(self):
        w = np.linspace(.1, 1.9, 60)
        a, b = original.fit(self.x, self.y), balanced.fit_weighted(self.x, self.y, w)
        self.assertEqual(a["mean"], b["mean"])
        self.assertEqual(a["scale"], b["scale"])
        self.assertEqual(b["threshold"], 0)
        self.assertEqual(b["lambda_l2"], 1)
        self.assertFalse(b["inference_group_label_used"])

    def test_constant_target_intercept_unpenalized(self):
        b = balanced.fit_weighted(self.x, np.ones(60)*.2, np.ones(60))
        np.testing.assert_allclose(original.predict(self.x, b), .2, atol=1e-12)

    def test_nonfinite_and_negative_rejected(self):
        for w in (np.full(60, np.nan), np.full(60, -1)):
            with self.assertRaises(AssertionError):
                balanced.fit_weighted(self.x, self.y, w)

    def test_deterministic_no_global_rng_change(self):
        before = np.random.get_state()
        a = balanced.fit_weighted(self.x, self.y, np.ones(60))
        self.assertEqual(a, balanced.fit_weighted(self.x, self.y, np.ones(60)))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
