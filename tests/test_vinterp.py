import unittest
import numpy as np

from teflow.mathext import vinterp, Metric


class TestVinterp(unittest.TestCase):
    def test_linear(self):
        xp = np.arange(0, 11, 1)
        yp = vinterp(
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            xp,
        )
        yref = 2 * xp
        self.assertTrue(np.allclose(yp, yref))

    def test_linear_ext(self):
        xp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yp = vinterp(
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            xp,
            method='linear',
            left=-404,
            right=None,
        )
        yref = [-404, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.assertTrue(np.allclose(yp, yref))

    def test_line_ext(self):
        xp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yp = vinterp(
            [1, 2, 4, 8],
            [2, 4, 8, 16],
            xp,
            method='line',
            left=-404,
            right=None,
        )
        yref = [-404, 2, 4, 6, 8, 10, 12, 14, 16, 16, 16]
        self.assertTrue(np.allclose(yp, yref))


class TestInterpGPR(unittest.TestCase):
    def model(self, x):
        return x * np.cos(x)

    def gradient(self, x):
        return np.cos(x) - x*np.sin(x)

    def test_interp_gpr(self):
        xp = np.linspace(0, 12, 121)
        z_real = self.model(xp)
        dz_real = self.gradient(xp)

        xi = np.arange(1, 12, 2)
        zi = self.model(xi)

        paras = dict(kernel='rbf', method='gpr', scale=2.5, regular=1E-3)
        zp = vinterp(xi, zi, xp, **paras)
        dzp = vinterp(xi, zi, xp, gradient=True, **paras)

        rmse_zp = Metric.RMSE(zp, z_real)
        rmse_dzp = Metric.RMSE(dzp, dz_real)
        # np.savetxt('data_gpr.txt', np.c_[xp, zp, z_real, dzp, dz_real],
        #            fmt='%.6f', header='zp z_real dzp dz_real')
        self.assertLess(rmse_zp, 0.2)
        self.assertLess(rmse_dzp, 0.4)

