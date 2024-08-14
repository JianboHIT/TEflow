import unittest
import numpy as np

from teflow.mathext import vinterp, Metric
from teflow.mathext import _kernel_rbf, _kernel_dfx


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

    def _test_kernel(self, kernel, scale=1.0):
        xp = np.linspace(0, 12, 121)
        z_real = self.model(xp)
        dz_real = self.gradient(xp)

        xi = np.arange(1, 12, 2)
        zi = self.model(xi)

        paras = dict(kernel=kernel, method='gpr', scale=scale, regular=1E-3)
        zp = vinterp(xi, zi, xp, **paras)
        dzp = vinterp(xi, zi, xp, gradient=True, **paras)

        rmse_zp = Metric.RMSE(zp, z_real)
        rmse_dzp = Metric.RMSE(dzp, dz_real)
        # np.savetxt(f'data_{kernel}.txt', np.c_[xp, zp, z_real, dzp, dz_real],
        #            fmt='%.6f', header='zp z_real dzp dz_real')
        # np.savetxt(f'data_{kernel}_gradient.txt', np.c_[xi, zi, self.gradient(xi)],
        #            fmt='%.6f', header='xi zi dzi')
        return rmse_zp, rmse_dzp

    def test_interp_rbf(self):
        # print('interp_rbf\ns rmse_zp rmse_dzp')
        # for s in np.arange(0.5, 4.1, 0.1):
        #     rmse_zp, rmse_dzp = self._test_kernel('rbf', scale=s)
        #     print(f'{s:g} {rmse_zp} {rmse_dzp}')
        rmse_zp, rmse_dzp = self._test_kernel('rbf', scale=2.5)
        self.assertLess(rmse_zp, 0.2)   # 0.17
        self.assertLess(rmse_dzp, 0.4)  # 0.32

    def test_interp_dfx(self):
        # print('interp_dfx\ns rmse_zp rmse_dzp')
        # for s in np.arange(0.5, 4.1, 0.1):
        #     rmse_zp, rmse_dzp = self._test_kernel('dfx', scale=s)
        #     print(f'{s:g} {rmse_zp} {rmse_dzp}')
        rmse_zp, rmse_dzp = self._test_kernel('dfx', scale=2.3)
        self.assertLess(rmse_zp, 0.3)   # 0.29
        self.assertLess(rmse_dzp, 0.6)  # 0.54

    def test_kernel_rbf(self):
        x = np.linspace(-5, 5, 11)
        y = _kernel_rbf(x, 0, 1.0)[0]
        yreal = np.exp(-x**2/2)
        dy = _kernel_rbf(x, 0, 1.0, gradient=True)[0]
        dyreal = -x*np.exp(-x**2/2)
        self.assertTrue(np.allclose(y, yreal))
        self.assertTrue(np.allclose(dy, dyreal))
        self.assertAlmostEqual(y.max(), 1.0)

    def test_kernel_dfx(self):
        x = np.linspace(-5, 5, 11)
        y = _kernel_dfx(x, 0, 1.0)[0]
        yreal = 4*np.exp(x)/(1+np.exp(x))**2
        dy = _kernel_dfx(x, 0, 1.0, gradient=True)[0]
        dyreal = 4*(np.exp(x)*(1+np.exp(x))-2*np.exp(2*x))/(1+np.exp(x))**3
        self.assertTrue(np.allclose(y, yreal))
        self.assertTrue(np.allclose(dy, dyreal))
        self.assertAlmostEqual(y.max(), 1.0)
