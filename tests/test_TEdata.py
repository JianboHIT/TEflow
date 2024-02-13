import unittest
import numpy as np

from teflow.loader import TEdatacol, TEdataset


class TestTEdatacol(unittest.TestCase):
    obj = TEdatacol
    q = 1.602176634
    seq = list(range(10))

    def test_TRSK(self):
        d = self.obj(data=self.seq, group='TCTSTK')
        v = np.squeeze(d.gget('TRSK'))
        r = np.array([4, 10000, 3, 5])
        self.assertTrue(np.allclose(v, r))

    def test_KR(self):
        d = self.obj(data=[1, 4, 250, 200, 10], group='TNUSK',)
        v = np.squeeze(d.gget('KR'))
        r = np.array([10, 10/self.q])
        self.assertTrue(np.allclose(v, r))

    def test_not_KR(self):
        d = self.obj(data=self.seq, group='TNSK')
        with self.assertRaisesRegex(ValueError, 'Failed .* R'):
            d.gget('KR')


class TestTEdataset(unittest.TestCase):
    def setUp(self):
        self.obj = TEdataset
        self.seq = list(range(10))

    def test_not_KR(self):
        d = self.obj(data=self.seq, group='TNSK')
        with self.assertRaisesRegex(ValueError, 'Failed .* R'):
            d.gget('KR')

    def test_mismatch(self):
        d = self.obj(data=self.seq, group='TNTUSK')
        with self.assertRaisesRegex(ValueError, 'Failed .* R'):
            d.gget('KR')
