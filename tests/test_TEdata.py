import unittest
import numpy as np

from teflow.loader import TEdataset


class TestTEdataset(unittest.TestCase):
    q = 1.602176634
    seq = list(range(10))

    def test_TRSK(self):
        d = TEdataset(data=self.seq, group='TCTSTK')
        v = np.squeeze(d.gget('TRSK'))
        r = np.array([4, 10000, 3, 5])
        self.assertTrue(np.allclose(v, r))

    def test_KR(self):
        d = TEdataset(data=[1, 4, 250, 200, 10], group='TNUSK',)
        v = np.squeeze(d.gget('KR'))
        r = np.array([10, 10/self.q])
        self.assertTrue(np.allclose(v, r))

    def test_not_KR(self):
        d = TEdataset(data=self.seq, group='TNSK')
        with self.assertRaisesRegex(ValueError, 'Failed .* R'):
            d.gget('KR')

    def test_not_KR_2(self):
        d = TEdataset(data=self.seq, group='TNSK', independent=False)
        with self.assertRaisesRegex(ValueError, 'Failed .* R'):
            d.gget('KR')

    def test_mismatch(self):
        d = TEdataset(data=self.seq, group='TNTUSK', independent=False)
        with self.assertRaisesRegex(ValueError, 'Failed .* R'):
            d.gget('KR')
