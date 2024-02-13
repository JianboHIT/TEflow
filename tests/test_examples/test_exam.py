import os
import runpy
import shutil
import unittest

import teflow.interface as cmd

from test_cmd.cmdio import arraycmp

PWD = os.path.dirname(__file__)

class BaseTestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_cwd = os.getcwd()
        os.chdir(PWD)
        if os.path.exists(cls.fdir):
            shutil.rmtree(cls.fdir)
        shutil.copytree(f'../../examples/{cls.fdir}', f'./{cls.fdir}')
        os.chdir(f'{PWD}/{cls.fdir}')
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_cwd)
        return super().tearDownClass()

    def setUp(self):
        print()
        return super().setUp()


class TestEngineeringPerf(BaseTestExamples):
    fdir = 'EngineeringPerf'
    def test_step_1_datas_n(self):
        # tef-engout datas_n.txt
        cmd.do_engout(['datas_n.txt',])
        fcmp = arraycmp('datas_n_engout.txt',
                        'datas_n_engout-check.txt',
                        atol=1E-3)
        self.assertTrue(fcmp)

    def test_step_2_datas_np(self):
        # tef-engout -R --pair datas_np.txt
        cmd.do_engout(['-R', '--pair', 'datas_np.txt'])
        fcmp = arraycmp('datas_np_engout.txt',
                        'datas_np_engout-check.txt',
                        atol=1E-3)
        self.assertTrue(fcmp)

class TestInstrumentRawFiles(BaseTestExamples):
    fdir = 'InstrumentRawFiles'
    def test_autoparse(self):
        for fn in os.listdir():
            cmd.do_format(['--auto-type', fn])

class TestKappaFitting(BaseTestExamples):
    fdir = 'KappaFitting'
    def test_KappaFitting_file(self):
        runpy.run_path('ConstrainedFitting.py')

class TestLorenzInterp(BaseTestExamples):
    fdir = 'LorenzInterp'
    def test_step_1_solve(self):
        # python3 cal_L.py
        runpy.run_path('cal_L.py')
        fcmp = arraycmp('Lorenz_number.txt',
                        'Lorenz_number-check.txt',
                        atol=2E-3)
        self.assertTrue(fcmp)

    def test_step_2_interp(self):
        # cp Seebeck-check.txt Seebeck.txt
        # tef-interp -m cubic Lorenz_number.txt Seebeck.txt
        shutil.copy('Seebeck-check.txt', 'Seebeck.txt')
        cmd.do_interp(['-m', 'cubic', 'Lorenz_number.txt', 'Seebeck.txt'])
        fcmp = arraycmp('Seebeck.txt',
                        'Seebeck_interp-check.txt',
                        atol=2E-3)
        self.assertTrue(fcmp)

class TestZTdev(BaseTestExamples):
    fdir = 'ZTdev'
    def test_step_1_by_data(self):
        # tef-ztdev Data_PbTe.txt
        cmd.do_ztdev(['Data_PbTe.txt',])
        fcmp = arraycmp('Data_PbTe_ztdev.txt',
                        'ZTdev_PbTe-check.txt',
                        atol=2E-3)
        self.assertTrue(fcmp)

    def test_step_2_by_yita(self):
        # tef-ztdev -y Yita_PbTe.txt
        cmd.do_ztdev(['-y', 'Yita_PbTe.txt'])
        fcmp = arraycmp('Yita_PbTe_ztdev.txt',
                        'ZTdev_PbTe-check.txt',
                        atol=2E-3)
        self.assertTrue(fcmp)
