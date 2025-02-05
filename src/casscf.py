from pyscf import gto, scf, mcscf, fci








# N2 at equilibrium geometry
b = 1.2
mol = gto.M(
    atom = 'N 0 0 0; N 0 0 %f'%b,
    basis = 'cc-pvdz',
)

mf = scf.RHF(mol).run()

# First use FCI solver
mc = mcscf.CASSCF(mf, 8, 6).run()

# Use SCI (Selected CI) to replace fcisolver
# mc = mcscf.CASSCF(mf, 8, 6)
# mc.fcisolver = fci.SCI(mol)
# mc.kernel()

# State average: the fix_spin_ enforces the spin for
# all states, otherwise the fcisolver.spin is not enough
# mc = mcscf.CASSCF(mf, 8, 6)
# mc.fcisolver = fci.direct_spin0.FCI(mol) # the spin0 is only singlet
# otherwise use the spin1, which is general
# mc = mc.state_average_([.25, .25, .25, .25])
# mc.fcisolver.spin = 0
# mc.fix_spin_(ss=0)
# mc.kernel()


#
# state-average over 1 triplet + 2 singlets
# Note direct_spin1 solver is called here because the CI solver will take
# spin-mix solution as initial guess which may break the spin symmetry
# required by direct_spin0 solver
#
# weights = np.ones(3)/3
# solver1 = fci.direct_spin1_symm.FCI(mol)
# solver1.spin = 2
# solver1 = fci.addons.fix_spin(solver1, shift=.2, ss=2)
# solver1.nroots = 1
# solver2 = fci.direct_spin0_symm.FCI(mol)
# solver2.spin = 0
# solver2.nroots = 2

# mc = mcscf.CASSCF(mf, 8, 8)
# mcscf.state_average_mix_(mc, [solver1, solver2], weights)

# You can also use UHF/UKS as the guess orbitals
# but the cas is still RHF-based
mol.spin = 2
mf = scf.UHF(mol).run()
mc = mcscf.CASSCF(mf, 8, 6)
mc.kernel()
# or use the UCASSCF to have an uestricted CAS
mc = mcscf.UCASSCF(mf, 8, (4,2))
mc.kernel()