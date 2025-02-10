
from pyscf import ao2mo

class NQSSolver(object):
    def __init__(self):
        self.mf = None
        self.nelec = None
        self.spin = None
        self.norb = None

    def kernel(self, h1e, eri, norb, nelec, **kwargs):
        # restore the full 4-index eri tensor
        eri_s1 = ao2mo.restore('s1', eri, norb)

        # here we actually compute the nqs energy
        e_nqs, ci_nqs = ...

        # not sure whether you can get the CI wfn

        return e_nqs, ci_nqs

    def make_rdm1(self, ci_vector, norb, nelec):
        D = np.zeros((norb, norb))
        # calculate the 1-RDM

        return D

    # def make_rdm2(self, ci_vector, norb, nelec):

    def spin_square(self, ci_vector, norb, nelec):
        return 0, 1