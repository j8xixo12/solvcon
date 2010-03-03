from unittest import TestCase

class TestArrangement(TestCase):
    def test_arrangement_registry(self):
        from ..case import BaseCase, BlockCase
        self.assertNotEqual(
            id(BaseCase.arrangements),
            id(BlockCase.arrangements)
        )

class TestCaseInfo(TestCase):
    def test_CaseInfo(self):
        from ..case import CaseInfo
        tdic = CaseInfo(_defdict={
            'lay1_1': '1_1',
            'lay1_2.lay2_1': '1_2/2_1',
            'lay1_2.lay2_2': '1_2/2_2',
            'lay1_3.': '1_3/',
        })
        self.assertEqual(tdic.lay1_1, '1_1')
        self.assertEqual(tdic.lay1_2.lay2_1, '1_2/2_1')
        self.assertEqual(tdic.lay1_2.lay2_2, '1_2/2_2')
        self.assertEqual(len(tdic.lay1_3), 0)

class TestBaseCase(TestCase):
    def test_empty_fields(self):
        from ..case import BaseCase
        case = BaseCase()
        # execution related.
        self.assertTrue(isinstance(case.execution.runhooks, list))
        self.assertEqual(case.execution.run_inner, False)
        self.assertEqual(case.execution.time, 0.0)
        self.assertEqual(case.execution.time_increment, 0.0)
        self.assertEqual(case.execution.step_init, 0)
        self.assertEqual(case.execution.step_current, None)
        self.assertEqual(case.execution.steps_run, None)
        self.assertEqual(case.execution.cCFL, 0.0)
        self.assertEqual(case.execution.aCFL, 0.0)
        self.assertEqual(case.execution.mCFL, 0.0)
        self.assertEqual(case.execution.neq, 0)
        self.assertEqual(case.execution.var, dict())
        self.assertEqual(case.execution.varstep, None)
        # io related.
        self.assertEqual(case.io.abspath, False)
        self.assertEqual(case.io.basedir, None)
        self.assertEqual(case.io.basefn, None)
        # condition related.
        self.assertEqual(case.condition.mtrllist, list())
        # solver related.
        self.assertEqual(case.solver.solvertype, None)
        self.assertEqual(case.solver.solverobj, None)
        # logging.
        self.assertEqual(case.log.time, dict())

    def test_abspath(self):
        import os
        from ..case import BaseCase
        case = BaseCase(basedir='.', abspath=True)
        path = os.path.abspath('.')
        self.assertEqual(case.io.basedir, path)

    def test_init(self):
        from ..case import BaseCase
        case = BaseCase()
        self.assertFalse(case._have_init)
        case.init()
        self.assertTrue(case._have_init)

from ..hook import Initializer, Calculator
class CaseInitSet(Initializer):
    _varnames_ = (
        # key, putback.
        ('soln', True,),
        ('dsoln', True,),
    )
    def _set_data(self, **kw):
        soln = kw['soln']
        dsoln = kw['dsoln']
        # solutions.
        soln.fill(0.0)
        dsoln.fill(0.0)
class CaseInitCollect(Initializer):
    def preloop(self):
        # super preloop.
        soln = self._collect_interior('soln')
        dsoln = self._collect_interior('dsoln')
        soln.fill(0.0)
        dsoln.fill(0.0)
        self._spread_interior(soln, 'soln')
        self._spread_interior(dsoln, 'dsoln')
class CaseCalc(Calculator):
    def postloop(self):
        self._collect_solutions()
del Initializer, Calculator

class TestBlockCaseRun(TestCase):
    time = 0.0
    time_increment = 1.0
    nsteps = 10

    def _get_case(self, CaseInit, **kw):
        from ..testing import get_blk_from_sample_neu
        from ..solver import BlockSolver
        from ..case import BlockCase
        case = BlockCase(basedir='.', basefn='blockcase',
            solvertype=BlockSolver, neq=1,
            steps_run=self.nsteps, time_increment=self.time_increment,
            **kw
        )
        case.info = lambda *a: None
        case.load_block = get_blk_from_sample_neu
        case.execution.runhooks.append(CaseInit(case))
        case.execution.runhooks.append(CaseCalc(case))
        case.init()
        return case

class TestSequential(TestBlockCaseRun):
    def test_soln_set(self):
        from numpy import zeros
        from ..domain import Domain
        case = self._get_case(CaseInitSet, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        soln = svr.soln[ngstcell:,0]
        # calculate reference
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += svr.clvol[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_set(self):
        from numpy import zeros
        from ..domain import Domain
        case = self._get_case(CaseInitSet, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        dsoln = svr.dsoln[ngstcell:,0,:]
        # calculate reference
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += svr.clcnd[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

    def test_soln_collect(self):
        from numpy import zeros
        from ..domain import Domain
        case = self._get_case(CaseInitCollect, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        soln = svr.soln[ngstcell:,0]
        # calculate reference
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += svr.clvol[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_collect(self):
        from numpy import zeros
        from ..domain import Domain
        case = self._get_case(CaseInitCollect, domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        dsoln = svr.dsoln[ngstcell:,0,:]
        # calculate reference
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += svr.clcnd[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

class TestParallel(TestBlockCaseRun):
    npart = 3

    def test_soln_set(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ..domain import Collective
        case = self._get_case(CaseInitSet,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_set(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ..domain import Collective
        case = self._get_case(CaseInitSet,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        dsoln = case.execution.var['dsoln'][:,0,:]
        # calculate reference
        blk = case.solver.domainobj.blk
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += blk.clcncrd*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

    def test_soln_collect(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ..domain import Collective
        case = self._get_case(CaseInitCollect,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())
    def test_dsoln_collect(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from ..domain import Collective
        case = self._get_case(CaseInitCollect,
            npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        dsoln = case.execution.var['dsoln'][:,0,:]
        # calculate reference
        blk = case.solver.domainobj.blk
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += blk.clcncrd*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())