from test_cmd.cmdio import BaseTestCmd
from teflow.interface import do_kappa


class TestCmdKappa(BaseTestCmd):
    fwork = __file__
    fdata = {
        'SnTe_data': ['SnTe_kappa.txt', 'SnTe_kappa-check.txt'],
        'SnTe_cumu': ['SnTe_kappa_cumu.txt', 'SnTe_kappa_cumu-check.txt'],
        'SnTe_rate': ['SnTe_kappa_rate.txt', 'SnTe_kappa_rate-check.txt'],
        'SnTe_spec': ['SnTe_kappa_spec.txt', 'SnTe_kappa_spec-check.txt'],
    }
    def test_kappa_SnTe(self):
        do_kappa(['SnTe.cfg'])
        self.assertTrue(self.fcomp('SnTe_data', atol=0.01))
        self.assertTrue(self.fcomp('SnTe_cumu', atol=0.01))
        self.assertTrue(self.fcomp('SnTe_rate', rtol=0.001))
        self.assertTrue(self.fcomp('SnTe_spec', rtol=0.001))
