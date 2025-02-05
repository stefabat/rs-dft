from pyscf import gto, scf, dft, fci, mcscf, ao2mo, tools
import numpy as np

if __name__ == '__main__':
    # hydrogen chain
    nH = 8

    mol = gto.M()
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=3
    mol.atom = [['H', (0, 0, 6*i)] for i in range(nH)]
    mol.unit = 'B'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    fcisolver = fci.FCI(mf)
    e, ci = fcisolver.kernel()
    dm = fcisolver.make_rdm1(ci, nH, (nH//2,nH//2))

    eigenvalues, _ = np.linalg.eigh(dm)
    print("Eigenvalues:", eigenvalues)
