import unittest

from memory_echo_abcd.dce import (
    CONTAMINATION_MARGIN,
    is_contaminated,
    legacy_sigmoid_dce,
    relative_distance_dce,
)


class RelativeDistanceDCETest(unittest.TestCase):
    def test_anchor_interpretation(self):
        self.assertEqual(relative_distance_dce(0.2, 1.0), 0.0)
        self.assertEqual(relative_distance_dce(1.0, 0.2), 1.0)
        self.assertEqual(relative_distance_dce(0.5, 0.5), 0.5)

    def test_is_bounded_and_monotonic(self):
        low = relative_distance_dce(-0.2, 0.8)
        high = relative_distance_dce(0.8, -0.2)
        self.assertTrue(0.0 <= low < 0.5 < high <= 1.0)

    def test_float_overshoot_is_clamped(self):
        self.assertEqual(relative_distance_dce(1.0000001, 0.0), 1.0)
        self.assertEqual(relative_distance_dce(0.0, 1.0000001), 0.0)

    def test_contamination_boundary_matches_legacy_v2(self):
        gold = 0.3
        echo = gold + CONTAMINATION_MARGIN
        self.assertAlmostEqual(legacy_sigmoid_dce(echo, gold), 0.60, places=12)
        self.assertTrue(is_contaminated(echo, gold))
        self.assertFalse(is_contaminated(echo - 1e-8, gold))


if __name__ == "__main__":
    unittest.main()
