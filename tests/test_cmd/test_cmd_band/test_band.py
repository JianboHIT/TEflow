from test_cmd.cmdio import BaseTestCmd
from teflow.interface import do_band

class TestCmdBand(BaseTestCmd):
    fwork = __file__
    fdata = {
        'SPB': ['TS_SPB.txt', 'TS_SPB-check.txt'],
        'SKB': ['TS_SKB.txt', 'TS_SKB-check.txt'],
    }
    def test_band_TS_SPB(self):
        do_band(['-g', 'TS', '-s', 'SPB', 'TS.txt'])
        self.assertTrue(self.fcomp('SPB', rtol=1E-3))

    def test_band_TS_SKB(self):
        do_band(['-g', 'TS', '--gap', '0.1', '-s', 'SKB', 'TS.txt'])
        self.assertTrue(self.fcomp('SKB', rtol=1E-3))
