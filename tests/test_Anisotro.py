import unittest
import numpy as np
import teflow.anisotro as tea


class TestTens2seq(unittest.TestCase):
    def setUp(self):
        self.original = [[1, 1, 4],
                         [3, 2, 1],
                         [4, 5, 4]]
        return super().setUp()

    def test_tens2seq_default(self):
        self.assertTrue(np.allclose(
            tea.tens2seq(self.original),
            [1, 2, 4, 3, 4, 2]
        ))

    def test_tens2seq_upper(self):
        self.assertTrue(np.allclose(
            tea.tens2seq(self.original, use_upper=True),
            [1, 2, 4, 1, 4, 1]
        ))

    def test_tens2seq_lower(self):
        self.assertTrue(np.allclose(
            tea.tens2seq(self.original, use_upper=False),
            [1, 2, 4, 5, 4, 3]
        ))


class TestSeq2tens(unittest.TestCase):
    def test_seq2tens(self):
        self.assertTrue(np.allclose(
            tea.seq2tens(1, 2, 4, 5, 6),
            [[1, 0, 6],
             [0, 2, 5],
             [6, 5, 4]]
        ))

class TestTensRotate(unittest.TestCase):
    def setUp(self):
        self.T = [1, 2, 4]
        self.R = tea.comb_rotations(np.pi/6, np.pi/4)
        return super().setUp()

    def test_rotate_example(self):
        Ty = tea.rotate3d(np.radians(30), *self.T, orient='y')
        Tyz = tea.rotate3d(np.radians(45), *Ty, orient=[0, 0, 1])   # z-axis
        Tr = tea.rotate3d(self.R, *self.T)
        self.assertTrue(np.allclose(Tyz, Tr), msg=f'\nNot equal:\n{Tyz}\n{Tr}')

    def test_project(self):
        vec = np.array([1,0,0]) @ self.R    # rotate vec with R
        phi = np.arccos(vec[2])             # angle between vec and z-axis
        theta = np.arctan2(vec[1]/np.sin(phi), vec[0]/np.sin(phi))  # angle in xy plane
        Txx = tea.project3d(theta, phi, *self.T)
        T11 = tea.rotate3d(self.R, *self.T)[0]
        self.assertAlmostEqual(Txx, T11)
