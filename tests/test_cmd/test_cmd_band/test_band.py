from test_cmd.cmdio import BaseTestCmd
from teflow.interface import do_band

class TestCmdBand(BaseTestCmd):
    fwork = __file__
    fdata = {
        'SPB': ['TS_SPB.txt', 'TS_SPB-check.txt'],
        'SKB': ['TS_SKB.txt', 'TS_SKB-check.txt'],
        'TB': ['double_band.txt', 'double_band-check.txt'],
        'TBsolver': ['double_solve.txt', 'double_solve-check.txt'],
    }
    def test_band_TS_SPB(self):
        do_band(['-g', 'TS', '-s', 'SPB', 'TS.txt'])
        self.assertTrue(self.fcomp('SPB', rtol=1E-3))

    def test_band_TS_SKB(self):
        do_band(['-g', 'TS', '--gap', '0.1', '-s', 'SKB', 'TS.txt'])
        self.assertTrue(self.fcomp('SKB', rtol=1E-3))

    def test_band_double(self):
        # tef-band -f --T '300 323:25:873' --EF 0 double.cfg
        do_band(['-f', '--T', '300 323:25:873', '--EF', '0', 'double.cfg'])
        self.assertTrue(self.fcomp('TB', rtol=1E-3))

    def test_band_solve(self):
        # tef-band -f --T '300 323:25:873' --EF '100 @ N' --initial -10 -s solve  double.cfg
        do_band(['-f', '--T', '300 323:25:873', '--EF', '100 @ N',
                 '--initial', '-10', '-s', 'solve', 'double.cfg'])
        self.assertTrue(self.fcomp('TBsolver', rtol=1E-3))
