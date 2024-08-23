import unittest

from teflow.bandlib import APSSPB, APSSKB, MultiBands


class TestMultiBands(unittest.TestCase):
    def setUp(self):
        self.mb1 = MultiBands([], [])
        self.mb2 = MultiBands([], [])
        self.band1 = APSSPB(sigma0=1)
        self.band2 = APSSPB(sigma0=10)
        self.band3 = APSSKB(sigma0=100, Eg=0.1)

    def test_append_valid_band(self):
        self.mb1.append(self.band1, 0.5, 'C')
        self.assertEqual(len(self.mb1.bands), 1)
        self.assertAlmostEqual(self.mb1.deltas[0], 0.5)
        self.assertEqual(self.mb1.bands[0], self.band1)
        self.assertEqual(self.band1._q_sign, -1)  # Conduction band

    def test_append_valence_band(self):
        self.mb1.append(self.band1, -0.3, 'V')
        self.assertEqual(len(self.mb1.bands), 1)
        self.assertAlmostEqual(self.mb1.deltas[0], -0.3)
        self.assertEqual(self.mb1.bands[0], self.band1)
        self.assertEqual(self.band1._q_sign, +1)  # Valence band

    def test_append_invalid_band(self):
        with self.assertRaises(ValueError):
            self.mb1.append('NotABand', 0.5)

    def test_append_invalid_delta(self):
        with self.assertRaises(ValueError):
            self.mb1.append(self.band1, 'not_a_float')

    def test_extend_other(self):
        # Prepare mb2 with some bands
        self.mb2.append(self.band2, 0.2, 'C')
        self.mb2.append(self.band3, -0.4, 'V')

        # Extend mb1 with mb2
        self.mb1.extend(self.mb2)

        self.assertEqual(len(self.mb1.bands), 2)
        self.assertAlmostEqual(self.mb1.deltas[0], 0.2)
        self.assertAlmostEqual(self.mb1.deltas[1], -0.4)
        self.assertEqual(self.mb1.bands[0], self.band2)
        self.assertEqual(self.mb1.bands[1], self.band3)
        self.assertEqual(self.band2._q_sign, -1)  # Conduction band
        self.assertEqual(self.band3._q_sign, +1)  # Valence band

    def test_extend_invalid_instance(self):
        with self.assertRaises(ValueError):
            self.mb1.extend('NotAMultiBandsInstance')
