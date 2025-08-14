from pyscf import gto, scf, mcscf, fci, dft
import numpy as np
from opt_einsum import contract

# Fock matrix and energy for a given density in AO basis
# for doubly occupied orbitals
def fock(D, g, h):
    # print('One-electron energy: ', np.einsum('ij,ij->', h, D))
    J = np.einsum('ijkl,kl->ij', g, D)        # Coulomb
    # print('Hartree energy: ',  .5*np.einsum('ij,ij->', J, D))
    K = -0.5*np.einsum('ilkj,kl->ij', g, D)   # Exchange
    # print('Exchange energy: ', .5*np.einsum('ij,ij->', K, D))
    if h is not None:
        F = h + J + K                             # Fock matrix
        E = .5*np.einsum('ij,ij->', h + F, D)
    else:
        F = J + K
        E = 0.0

    return F, E

# Generalized Fock matrix and energy for a given density in MO basis
def fock_gen(h, g, D1, D2):
    F_eff  = contract('pr,rq->pq', D1, h)
    F_eff += 2.0*np.einsum('prst,rqst->pq', D2, g)
    return F_eff

# compute AO to MO transformation of two-electron integrals
# for the given set of MO coefficients C
def eri_ao2mo(C, g):
    pqrw = np.einsum("pqrs,sw->pqrw", g   , C)
    pqvw = np.einsum("pqrw,rv->pqvw", pqrw, C)
    puvw = np.einsum("pqvw,qu->puvw", pqvw, C)
    tuvw = np.einsum("puvw,pt->tuvw", puvw, C)
    return puvw, tuvw

def update_total_density(C_act, D_ao_inactive, D_mo_active):
    D_ao_active = np.einsum("pt, qu, tu->pq", C_act, C_act, D_mo_active)
    D_ao_total = D_ao_active + D_ao_inactive
    return D_ao_total


# active is either a single list of active orbitals or a tuple with two lists
# the active orbital indices are 1-based
def casscf(mf, nel_act, nmo_act, active, max_iter=100, threshold=1e-6, alpha=None, fci_solver=None, debug=False):

    # determine if the calculation is restricted or unrestricted
    if type(mf) == scf.hf.RHF or type(mf) == dft.ks.RKS:
        nspins = 1
    elif type(mf) == scf.hf.UHF or type(mf) == dft.ks.UKS:
        nspins = 2
    else:
        raise TypeError('unsupported mf object')

    # check format of active orbitals
    if type(active) == list:
        # then only one set of active orbitals
        assert(len(active) == nmo_act)
        active = [i-1 for i in active]
        if nspins == 2:
            active = (active, active)
    elif type(active) == tuple:
        assert(len(active) == 2)
        assert(len(active[0]) == nmo_act)
        assert(len(active[1]) == nmo_act)
        active = ([i-1 for i in active[0]],[i-1 for i in active[1]])
    else:
        raise TypeError('active orbitals must be a list or tuple')

    # we only support an even number of inactive electrons
    nel_ina = mf.mol.nelectron - nel_act
    assert(nel_ina % 2 == 0)
    nmo_ina = nel_ina // 2
    nbas = mf.mol.nao
    spin = mf.mol.spin
    assert(nel_act % 2 == spin % 2)
    mult = mf.mol.multiplicity
    nel_fci = ((nel_act + mult - 1)//2, (nel_act - mult + 1)//2)

    # check if it is a range-separated calculation
    if hasattr(mf, 'omega'):
        omega = mf.omega
    else:
        omega = None

    # determine the inactive orbitals
    if nspins == 1:
        occupied = np.nonzero(mf.mo_occ)[0]
        inactive = [i for i in occupied if i not in active]
    elif nspins == 2:
        inactive = []
        for ispin in range(nspins):
            occupied = np.nonzero(mf.mo_occ[ispin])[0]
            inactive.append([i for i in occupied if i not in active[ispin]])
        # and make it a tuple
        inactive = tuple(inactive)
        assert(len(inactive[0]) == len(inactive[1]))

    # define inactive and active orbital spaces
    if nspins == 1:
        print('Inactive orbitals:', np.array(inactive)+1)
        print('Active orbitals:', np.array(active)+1)
    elif nspins == 2:
        print('Inactive orbitals alpha:', np.array(inactive[0])+1)
        print('Inactive orbitals beta: ', np.array(inactive[1])+1)
        print('Active orbitals alpha:', np.array(active[0])+1)
        print('Active orbitals beta: ', np.array(active[1])+1)

    # set occupation vectors for inactive and active MOs
    mo_occ_inactive = np.zeros_like(mf.mo_occ)
    if nspins == 1:
        mo_occ_inactive[inactive] = mf.mo_occ[inactive]
    else:
        mo_occ_inactive[0,inactive[0]] = mf.mo_occ[0,inactive[0]]
        mo_occ_inactive[1,inactive[1]] = mf.mo_occ[1,inactive[1]]

    print('Inactive occupations:\n', mo_occ_inactive)
    mo_occ_active = mf.mo_occ - mo_occ_inactive
    print('Active occupations:\n', mo_occ_active)

    # get inactive and active MO coefficients
    C = mf.mo_coeff
    if nspins == 1:
        C_I = C[:, inactive]
        C_A = C[:, active]
    elif nspins == 2:
        C_A = np.zeros((nspins, mf.mol.nao, nmo_act))
        C_A[0] = C[0,:,active[0]].transpose()
        C_A[1] = C[1,:,active[1]].transpose()

    # inactive density matrix in AO basis
    D_I_ao = mf.make_rdm1(C, mo_occ_inactive)
    # total density matrices in AO basis
    D1_ao = mf.make_rdm1(C)
    D2_ao = mf.make_rdm2(C)

    # transform them to MO basis, need to use the overlap!
    S = mf.get_ovlp()
    SC = S@C
    # D1_mo = C.T@S.T@D1_ao@S@C
    D1_mo = contract('λp,λδ,δq->pq', SC, D1_ao, SC)
    D2_mo = contract('λp,δq,λδστ,σr,τs->pqrs', SC, SC, D2_ao, SC, SC)

    # in the initial step, the active density matrix is equal to the
    # HF density matrix within the active subspace
    if nspins == 1:
        D_A_mo = np.zeros((nmo_act,nmo_act))
        for i in range(np.count_nonzero(mo_occ_active)):
            D_A_mo[i,i] = 2.0
    else:
        D_A_mo = np.zeros((nspins,nmo_act,nmo_act))
        for ispin in range(nspins):
            for i in range(np.count_nonzero(mo_occ_active[ispin])):
                D_A_mo[ispin,i,i] = 1.0


    # get kinetic energy integrals
    h = mf.get_hcore()
    # get lr two-body MO integrals
    # neri = nmo_act*(nmo_act+1)//2
    with mf.mol.with_range_coulomb(omega=omega):
        # if nspins == 1:
        g = mol.intor('int2e')
        # else:
            # g = np.zeros((3, neri, neri))
            # g[0] = mf.mol.ao2mo(C_A[0])
            # g[1] = mf.mol.ao2mo((C_A[0], C_A[0], C_A[1], C_A[1]))
            # g[2] = mf.mol.ao2mo(C_A[1])

    # get nuclear repulsion energy
    E_n = mf.mol.get_enuc()

    # create the FCI solver
    if fci_solver is None:
        if nspins == 1:
            fci_solver = fci.direct_spin1.FCI()
        else:
            fci_solver = fci.direct_uhf.FCI()
    # fci_solver = fci.addons.fix_spin_(fci_solver, ss=spin)

    # initialize history lists
    D_A_mo_history = []
    E_history = []

    n_iter = 0
    D_A_mo_history.append(D_A_mo)
    E_history.append(mf.e_tot)
    converged = False
    # start the iterative loop
    print(f'\nIter    <Ψ|S^2|Ψ>    Corr. energy      Total energy          |G|')
    while n_iter < max_iter:
        n_iter += 1

        F_I_ao, E_I = fock(D_I_ao, g, h)
        F_I_mo = contract('λt,λδ,δu->tu', C_A, F_I_ao, C_A)

        g_λuvw, g_tuvw = eri_ao2mo(C_A, g)

        # update total density matrix
        # D = D_I + D_A
        # # get the KS potential and LR active components to subtract
        # F_ks = mf.get_fock(dm=D)   # this is ks_mat

        # J_A_lr, K_A_lr = mf.get_jk(dm=D_A, omega=omega)

        # if nspins == 1:
        #     V_emb = F_ks - (J_A_lr - 0.5*K_A_lr)  # J_A_lr - 0.5*K_A_lr is ks_ref
        # else:
        #     # sum alpha and beta Coulopmb contributions to each other
        #     J_A_lr[:] += J_A_lr[::-1]
        #     V_emb = (F_ks - (J_A_lr - K_A_lr))

        # # transform embedding potential to active MO basis
        # if nspins == 1:
        #     V_emb_mo = np.einsum('pt,pq,qu->tu', C_A, V_emb, C_A)
        # else:
        #     V_emb_mo = np.zeros((2, nmo_act, nmo_act))
        #     for ispin in range(nspins):
        #         V_emb_mo[ispin] = np.einsum('pt,pq,qu->tu', C_A[ispin], V_emb[ispin], C_A[ispin])
        #         if debug:
        #             print(f'V_emb_mo[{ispin}] =\n', V_emb_mo[ispin])

        # # compute inactive energy
        # E_I = mf.energy_tot(dm=D) - E_n
        # if nspins == 1:
        #     E_I -=    np.einsum('pq,pq->', D_A, F_ks)
        #     E_I += .5*np.einsum('pq,pq->', D_A, (J_A_lr - 0.5*K_A_lr))
        # else:
        #     for ispin in range(nspins):
        #         E_I -=    np.einsum('pq,pq->', D_A[ispin], F_ks[ispin])
        #         E_I += .5*np.einsum('pq,pq->', D_A[ispin], (J_A_lr[ispin] - K_A_lr[ispin]))

        # solve the FCI problem (with the embedding potential and lr ERI)
        E_A, CI_vec = fci_solver.kernel(h1e=F_I_mo, eri=g_tuvw, norb=nmo_act, nelec=nel_fci)
        if debug:
            print('\nInactive energy:', E_I + E_n)
            print('Active energy:', E_A)

        # new active density matrix
        if nspins == 1:
            D_A_mo, d2_A_mo = fci_solver.make_rdm12(CI_vec, norb=nmo_act, nelec=nel_fci)
            # density damping
            # if alpha is not None:
            #     D_A_mo = (1-alpha)*D_A_mo + alpha*D_A_history[-1]
            # transform D_AO_mo to AO basis
            D_A_ao = contract('λt,tu,δu->λδ', C_A, D_A_mo, C_A)

            SS, Ms = fci.spin_square(CI_vec, norb=nmo_act, nelec=nel_fci)
        else:
            # D_A_mo = fci_solver.make_rdm1s(CI_vec, norb=nmo_act, nelec=nel_fci)
            D_A_mo, d2_A_mo = fci_solver.make_rdm12s(CI_vec, norb=nmo_act, nelec=nel_fci)
            if alpha is not None:
                D_A_mo_a = (1-alpha)*D_A_mo[0] + alpha*D_A_mo_history[-1][0]
                D_A_mo_b = (1-alpha)*D_A_mo[1] + alpha*D_A_mo_history[-1][1]
                D_A_mo = (D_A_mo_a, D_A_mo_b)
            # transform D_AO_mo to AO basis
            # for ispin in range(nspins):
                # D_A[ispin] = np.einsum('pt,tu,qu->pq', C_A[ispin], D_A_mo[ispin], C_A[ispin])

            SS, Ms = fci.spin_op.spin_square_general(*D_A_mo, *d2_A_mo, C_A)

        # add to history
        D_A_mo_history.append(D_A_mo)
        if debug:
            if nspins == 1:
                print('\nActive density matrix:\n', D_A_mo)
            else:
                print('\nAlpha active density matrix:\n', D_A_mo[0])
                print('\nBeta active density matrix:\n', D_A_mo[1])
                occ_a,mos_a = np.linalg.eigh(D_A_mo[0])
                occ_b,mos_b = np.linalg.eigh(D_A_mo[1])
                print('\nAlpha active orbitals:\n', mos_a[:,0])
                print('\nAlpha active orbitals:\n', mos_a)
                print('\nBeta active orbitals:\n', mos_b)
                print('Occupation numbers:\n', np.asarray([occ_a, occ_b]), '\n')
                print('Total number of el:\n', np.sum(np.asarray([occ_a, occ_b]),axis=1), '\n')

        # compute active Fock matrix in AO basis
        F_A_ao, _ = fock(D_A_ao, g, None)

        # effective Fock matrix
        F_eff_mo = np.zeros_like(F_A_ao)
        # print('F_eff_mo:', F_eff_mo)
        F_eff_mo[:nmo_ina,:] = contract('λi,λδ,δq->iq', C_I, 2*F_I_ao + F_A_ao, C) # Fiq
        # print('F_eff_mo:', F_eff_mo)
        F_eff_mo[nmo_ina:nmo_ina+nmo_act,:] = contract('λt,λδ,δσ,σq->tq', C_A, D_A_ao, F_I_ao, C) # Ftq (1)
        # print('F_eff_mo:', F_eff_mo)
        F_eff_mo[nmo_ina:nmo_ina+nmo_act,:] += contract('tuvw,λuvw,λq->tq', 2.0*d2_A_mo, g_λuvw, C) # Ftq (2)
        # print('F_eff_mo:', F_eff_mo)

        # form the gradient
        G = 2 * (F_eff_mo - np.transpose(F_eff_mo))
        print('G:', G)

        # check convergence
        e_vec = np.reshape(G, -1)
        error = np.linalg.norm(e_vec)

        # E_history.append(E_I + E_A + E_n)
        # delta_e = E_history[-1] - E_history[-2]
        # E_corr = E_history[-1] - E_history[0]
        E_corr = 0.0
        print(f'{n_iter:4d}    {SS:7.2f}     {E_corr:12.10f}    {E_history[-1]:12.10f}   {error:+10.2e}')

        converged = np.abs(error) < threshold
        if converged:
            print('Converged!')
            break

        #Extract some diagonals
        diag1 = np.einsum("pq,pm,qm->m", 2*F_I_ao+F_A_ao, C, C) # Diagonal of 2*Fin+Fact in MO basis
        diag2 = np.diagonal(F_eff_mo) #Diagonal of the effective Fock matrix

        #Form Hessian diagonal
        Hess = np.zeros((nbas,nbas))
        # Hess[:nmo_ina,nmo_ina:] = 2* diag1[nmo_ina:] - 2* diag1[:nmo_ina].reshape(-1,1) #Sum of a line and column vectors
        # Hess[nmo_ina:nmo_ina+nmo_act,:] = - 2 * diag2[nmo_ina:nmo_ina+nmo_act].reshape(-1, 1)
        # Hess[nmo_ina:nmo_ina+nmo_act,nmo_ina+nmo_act:] += np.einsum('tt,a->ta',D_A_ao,diag1[nmo_ina+nmo_act:])
        # Hess[nmo_ina:nmo_ina+nmo_act,:nmo_ina] += np.einsum('tt,a->ta',D_A_ao,diag1[:nmo_ina])
        # Hess += np.transpose(Hess)
        # Hess[:nmo_ina,:nmo_ina] = 1 #To avoid division by 0
        # Hess[nmo_ina + nmo_act:, nmo_ina + nmo_act:] = 1 #To avoid division by 0

        X = G / Hess
        expX = sp.linalg.expm(X)
        C = np.matmul(C,expX)


    # out of the iterative loop
    if not converged:
        print('Not converged!')

    # return in any case the last density matrix and energy
    return E_history[-1], D_A_mo_history[-1]






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