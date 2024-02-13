import os
import unittest
import numpy as np


def arraycmp(fout, fref, atol=0, rtol=0):
    dataout = np.loadtxt(fout, unpack=True, ndmin=2)
    dataref = np.loadtxt(fref, unpack=True, ndmin=2)
    return np.allclose(dataout, dataref, atol=atol, rtol=rtol)

class BaseTestCmd(unittest.TestCase):
    # require attribute:
    #
    #   fwork = __file__
    #
    #   fdata = {
    #       'SPB': ['TS_SPB.txt', 'TS_SPB-check.txt'],
    #       'SKB': ['TS_SKB.txt', 'TS_SKB-check.txt'],
    #   }
    @classmethod
    def setUpClass(cls):
        # change cwd
        cls.original_cwd = os.getcwd()
        os.chdir(os.path.dirname(cls.fwork))

        # remove existed output files
        for fout, *_ in cls.fdata.values():
            if os.path.exists(fout):
                os.remove(fout)
        return super().setUpClass()

    def setUp(self):
        # change to new line
        print()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_cwd)
        return super().tearDownClass()

    def fcomp(self, key, atol=0, rtol=0):
        fout, fref = self.fdata[key]
        return arraycmp(fout, fref, atol=atol, rtol=rtol)
