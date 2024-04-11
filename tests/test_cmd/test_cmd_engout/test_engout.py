from test_cmd.cmdio import BaseTestCmd
from teflow.interface import do_engout

class TestCmdEngout(BaseTestCmd):
    fwork = __file__
    fdata = {
        'single': ['TCSK_engout.txt', 'TCSK_engout-check.txt'],
    }
    def test_engout_single(self):
        do_engout(['TCSK.txt',])
        self.assertTrue(self.fcomp('single', atol=1E-3))
