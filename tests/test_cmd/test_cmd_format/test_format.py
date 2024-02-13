from test_cmd.cmdio import BaseTestCmd
from teflow.interface import do_format


class TestCmdFormat(BaseTestCmd):
    fwork = __file__
    fdata = {
        'TCSTK': ('TCSTK_format.txt', 'TCSTK_format-check.txt',),
    }
    def test_format_TCSTK(self):
        do_format(['-cg', 'TCSTK', 'TCSTK.txt'])
        self.assertTrue(self.fcomp('TCSTK', rtol=1E-3))

