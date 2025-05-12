import unittest
import numpy as np

from teflow.bandlib import APSSPB, APSSKB, RSPB, kB_eV


class TestLxS(unittest.TestCase):
    def setUp(self):
        self.m_d = 1
        self.sigma0 = 1000
        self.Eg = 0.2
        self.Kmass = 1
        self.EF = [[-0.1], [-0.05], [0], [0.05], [0.1], [0.15]]
        self.T = [300, 400, 500, 600, 700, 800, 900, 1000]

    def test_LxS_spb(self):
        spb = APSSPB(m_d=self.m_d, sigma0=self.sigma0, Kmass=self.Kmass)
        dataS = spb.S(self.EF, self.T)
        dataL = spb.L(self.EF, self.T)
        predL = RSPB.LxS(dataS, S_factor=RSPB.S0, L_factor=RSPB.L0)
        with self.assertRaises(AssertionError,
            msg="The two arrays are too close to be distinguished."):
            np.testing.assert_allclose(dataL, predL, rtol=1e-5)
        np.testing.assert_allclose(dataL, predL, rtol=0.02)

    def test_LxS_spb_ks(self):
        spb = APSSPB(m_d=self.m_d, sigma0=self.sigma0, Kmass=self.Kmass)
        dataS = spb.S(self.EF, self.T)
        dataL = spb.L(self.EF, self.T)
        predL = RSPB.LxS(dataS, S_factor=RSPB.S0, L_factor=RSPB.L0, dt=np.inf)
        with self.assertRaises(AssertionError,
            msg="The two arrays are too close to be distinguished."):
            np.testing.assert_allclose(dataL, predL, rtol=1e-5)
        np.testing.assert_allclose(dataL, predL, rtol=0.02)

    def test_LxS_skb(self):
        skb = APSSKB(m_d=self.m_d, sigma0=self.sigma0, Eg=self.Eg, Kmass=self.Kmass)
        dataS = skb.S(self.EF, self.T)
        dataL = skb.L(self.EF, self.T)
        dt = [self.Eg / (kB_eV * T) for T in self.T]
        predL = RSPB.LxS(dataS, S_factor=RSPB.S0, L_factor=RSPB.L0, dt=dt)
        with self.assertRaises(AssertionError,
            msg="The two arrays are too close to be distinguished."):
            np.testing.assert_allclose(dataL, predL, rtol=1e-5)
        np.testing.assert_allclose(dataL, predL, rtol=0.03)
