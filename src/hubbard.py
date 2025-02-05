import numpy
from pyscf import gto, scf, ao2mo, mcscf, fci


'''
User-defined Hamiltonian for CASSCF module.

Defining Hamiltonian once for SCF object, the derivate post-HF method get the
Hamiltonian automatically.
'''

mol = gto.M()
mol.nelectron = 8

# incore_anyway=True ensures the customized Hamiltonian (the _eri attribute)
# to be used.  Without this parameter, the MO integral transformation may
# ignore the customized Hamiltonian if memory is not enough.
mol.incore_anyway = True

#
# 1D anti-PBC Hubbard model at half filling
#
n = 9

h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = 1.0

h1[n-1,0] = h1[0,n-1] = 1.0
eigenvalues, eigenvectors = numpy.linalg.eigh(h1)
print("Eigenvalues of h1:", eigenvalues)

U = 8.0
eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = U

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = ao2mo.restore(8, eri, n)
mf.init_guess = '1e'
mf.kernel()
print(mf.mo_energy)

mf.CISD().run()

mf.CCSD().run()

fci.FCI(mf).run()

mc = mcscf.CASSCF(mf, 4, 4)
mc.kernel()
