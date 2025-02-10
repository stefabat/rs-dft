import numpy as np
from pyscf import gto, scf, cc, ao2mo, mcscf
# define a custom active space solver based on whatever method you like

class CCSDSolver(object):
    def __init__(self):
        self.mycc = None

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        fakemol = gto.M(verbose=0)
        nelec = np.sum(nelec)
        fakemol.nelectron = nelec
        fake_hf = scf.RHF(fakemol)
        fake_hf._eri = ao2mo.restore(8, eri, norb)
        fake_hf.get_hcore = lambda *args: h1e
        fake_hf.get_ovlp = lambda *args: np.eye(norb)
        fake_hf.kernel()
        self.mycc = cc.CCSD(fake_hf)
        eris = self.mycc.ao2mo()
        e_corr, t1, t2 = self.mycc.kernel(eris=eris)
        l1, l2 = self.mycc.solve_lambda(t1, t2, eris=eris)
        e_tot = self.mycc.e_tot + ecore
        return e_tot, CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])

    def make_rdm1(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        dm1 = self.mycc.make_rdm1(t1, t2, l1, l2, ao_repr=True)
        return dm1

    def make_rdm12(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        dm2 = self.mycc.make_rdm2(t1, t2, l1, l2, ao_repr=True)
        return self.make_rdm1(fake_ci, norb, nelec), dm2

    def spin_square(self, fake_ci, norb, nelec):
        return 0, 1

class CCSDAmplitudesAsCIWfn:
    def __init__(self, cc_amplitues):
        self.cc_amplitues = cc_amplitues

if __name__ == '__main__':
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RHF(mol).run()
    norb = 8
    nelec = 8
    mc = mcscf.CASCI(mf, norb, nelec)
    mc.fcisolver = CCSDSolver()
    mc.kernel()