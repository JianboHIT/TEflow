import unittest
import numpy as np

from teflow.mathext import vinterp


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
        
        