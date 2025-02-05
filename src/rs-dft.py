from pyscf import gto, scf, dft, fci, mcscf, ao2mo, tools
import numpy as np
from opt_einsum import contract
np.set_printoptions(precision=8, suppress=True, linewidth=200)

# active is either a single list of active orbitals or a tuple with two lists
# the active orbital indices are 1-based
def fci_srdft(mf, nel_act, nmo_act, active, max_iter=100, threshold=1e-6, alpha=None, fci_solver=None, debug=False):

    # determine if the calculation is restricted or unrestricted
    if type(mf) == scf.rhf.RHF or type(mf) == dft.rks.RKS:
        nspins = 1
    elif type(mf) == scf.uhf.UHF or type(mf) == dft.uks.UKS:
        nspins = 2
    else:
        raise TypeError('unsupported mf object')

    # check format of active orbitals
    if type(active) == list:
        # then only one set of active orbitals
        assert(len(active) == nmo_act)
        active = [i -1 for i in active]
        if nspins == 2:
            active = (active, active)
    elif type(active) == tuple:
        # the number of active orbitals must be the same for both spins
        assert(len(active) == 2)
        assert(len(active[0]) == nmo_act)
        assert(len(active[1]) == nmo_act)
        active = ([i-1 for i in active[0]],[i-1 for i in active[1]])
    else:
        raise TypeError('active orbitals must be a list or tuple')

    # we only support an even number of inactive electrons
    nel_ina = mf.mol.nelectron - nel_act
    assert(nel_ina % 2 == 0)
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
        nmo_ina = len(inactive)
    elif nspins == 2:
        inactive = []
        for ispin in range(nspins):
            occupied = np.nonzero(mf.mo_occ[ispin])[0]
            inactive.append([i for i in occupied if i not in active[ispin]])
        # and make it a tuple
        inactive = tuple(inactive)
        assert(len(inactive[0]) == len(inactive[1]))
        nmo_ina = len(inactive[0])

    # set some other important variables
    nao = mf.mol.nao

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

    # get active MO coefficients
    C = mf.mo_coeff
    if nspins == 1:
        C_A = C[:, active]
    elif nspins == 2:
        C_A = np.zeros((nspins, nao, nmo_act))
        C_A[0] = C[0,:,active[0]].transpose()
        C_A[1] = C[1,:,active[1]].transpose()
        # C_A = C[:,:,active]

    # inactive density matrix in AO basis
    D_I = mf.make_rdm1(C, mo_occ_inactive)

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

    # transform 1-RDM to AO basis
    D_A = np.zeros_like(D_I)
    if nspins == 1:
        D_A = contract('pt,tu,qu->pq', C_A, D_A_mo, C_A)
        # D_A = np.einsum('pt,tu,qu->pq', C_A, D_A_mo, C_A)
        # D_A = mf.make_rdm1(C, mo_occ_active) # same as above
    else:
        for ispin in range(nspins):
            D_A[ispin] = contract('pt,tu,qu->pq', C_A[ispin], D_A_mo[ispin], C_A[ispin])
            # D_A[ispin] = np.einsum('pt,tu,qu->pq', C_A[ispin], D_A_mo[ispin], C_A[ispin])

    # get lr two-body MO integrals
    neri = nmo_act*(nmo_act+1)//2
    with mf.mol.with_range_coulomb(omega=omega):
        if nspins == 1:
            g_lr_mo = mf.mol.ao2mo(C_A)
        else:
            g_lr_mo = np.zeros((3, neri, neri))
            g_lr_mo[0] = mf.mol.ao2mo(C_A[0])
            g_lr_mo[1] = mf.mol.ao2mo((C_A[0], C_A[0], C_A[1], C_A[1]))
            g_lr_mo[2] = mf.mol.ao2mo(C_A[1])

    # save full tensor
    with mol.with_range_coulomb(omega= omega):
        g_lr_ao = mol.intor('int2e')
    g_lr_mo_full = eri_ao2mo(C_A, g_lr_ao)
    print(g_lr_mo_full.shape)
    np.save('g_lr_mo.npy', g_lr_mo_full)
    # investigate sparsity of 2-body integrals
    threshold = 1e-12
    print(f"Non-zero elements g_lr_mo: {compute_sparsity(g_lr_mo_full, threshold):.2f}%")


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
    D_A_history = []
    E_history = []

    n_iter = 0
    D_A_history.append(D_A_mo)
    E_history.append(mf.e_tot)
    converged = False
    # start the iterative loop
    print(f'\nIter    <Ψ|S^2|Ψ>    Corr. energy      Total energy       Change')
    while n_iter < max_iter:
        n_iter += 1

        # update total density matrix
        D = D_I + D_A
        # get the KS potential and LR active components to subtract
        F_ks = mf.get_fock(dm=D)   # this is ks_mat

        J_A_lr, K_A_lr = mf.get_jk(dm=D_A, omega=omega)

        if nspins == 1:
            V_emb = F_ks - (J_A_lr - 0.5*K_A_lr)  # J_A_lr - 0.5*K_A_lr is ks_ref
        else:
            # sum alpha and beta Coulopmb contributions to each other
            J_A_lr[:] += J_A_lr[::-1]
            V_emb = (F_ks - (J_A_lr - K_A_lr))

        # transform embedding potential to active MO basis
        if nspins == 1:
            V_emb_mo = np.einsum('pt,pq,qu->tu', C_A, V_emb, C_A)
        else:
            V_emb_mo = np.zeros((2, nmo_act, nmo_act))
            for ispin in range(nspins):
                V_emb_mo[ispin] = np.einsum('pt,pq,qu->tu', C_A[ispin], V_emb[ispin], C_A[ispin])
                if debug:
                    print(f'V_emb_mo[{ispin}] =\n', V_emb_mo[ispin])

        # compute inactive energy
        E_I = mf.energy_tot(dm=D) - E_n
        if nspins == 1:
            E_I -=    np.einsum('pq,pq->', D_A, F_ks)
            E_I += .5*np.einsum('pq,pq->', D_A, (J_A_lr - 0.5*K_A_lr))
        else:
            for ispin in range(nspins):
                E_I -=    np.einsum('pq,pq->', D_A[ispin], F_ks[ispin])
                E_I += .5*np.einsum('pq,pq->', D_A[ispin], (J_A_lr[ispin] - K_A_lr[ispin]))

        # store 1-body and 2-body integrals
        np.save('V_emb_mo.npy', V_emb_mo)

        # solve the FCI problem with the embedding potential and lr ERI
        E_A, CI_vec = fci_solver.kernel(h1e=V_emb_mo, eri=g_lr_mo, norb=nmo_act, nelec=nel_fci)
        log_ci_states(fci_solver, thresh=1e-1)
        if debug:
            print('\nInactive energy:', E_I + E_n)
            print('Active energy:', E_A)
            # log_ci_states(fci_solver, thresh=1e-3)

        # new active density matrix
        if nspins == 1:
            D_A_mo = fci_solver.make_rdm1(CI_vec, norb=nmo_act, nelec=nel_fci)
            if alpha is not None:
                D_A_mo = (1-alpha)*D_A_mo + alpha*D_A_history[-1]
            # transform D_AO_mo to AO basis
            D_A = np.einsum('pt,tu,qu->pq', C_A, D_A_mo, C_A)

            # SS, Ms = fci.spin_square(CI_vec, norb=nmo_act, nelec=nel_fci)
            SS = 0
        else:
            # D_A_mo = fci_solver.make_rdm1s(CI_vec, norb=nmo_act, nelec=nel_fci)
            D_A_mo, d2_A_mo = fci_solver.make_rdm12s(CI_vec, norb=nmo_act, nelec=nel_fci)
            if alpha is not None:
                D_A_mo_a = (1-alpha)*D_A_mo[0] + alpha*D_A_history[-1][0]
                D_A_mo_b = (1-alpha)*D_A_mo[1] + alpha*D_A_history[-1][1]
                D_A_mo = (D_A_mo_a, D_A_mo_b)
            # transform D_AO_mo to AO basis
            for ispin in range(nspins):
                D_A[ispin] = np.einsum('pt,tu,qu->pq', C_A[ispin], D_A_mo[ispin], C_A[ispin])

            SS, Ms = fci.spin_op.spin_square_general(*D_A_mo, *d2_A_mo, C_A)

        # add to history
        D_A_history.append(D_A_mo)
        if debug:
            if nspins == 1:
                print('\nActive density matrix:\n', D_A_mo)
                print('Occupation numbers:\n', np.linalg.eigh(D_A_mo)[0], '\n')
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


        # orbital optimization step
        # Feff = np.zeros((nao,nao))
        # Feff[:n,:] = np.einsum("pq,pt,qn->tn", 2*Fin+Fact, C[:, :nIn], C) #Fiq
        # Feff[nIn:nIn+nAct,:] = np.einsum("tu,pu,pq,qn->tn", Dact, Cact, Fin, C) #Ftq (1)
        # Feff[nIn:nIn+nAct,:] += np.einsum("quvw,tuvw,qn->tn", puvw, D2act, C) #Ftq (2)


        # check convergence
        E_history.append(E_I + E_A + E_n)
        delta_e = E_history[-1] - E_history[-2]
        E_corr = E_history[-1] - E_history[0]
        print(f'{n_iter:4d}    {SS:7.2f}     {E_corr:12.10f}    {E_history[-1]:12.10f}   {delta_e:+10.2e}')

        converged = np.abs(delta_e) < threshold
        if converged:
            print('Converged!')
            break

    if not converged:
        print('Not converged!')

    # return in any case the last density matrix and energy
    return E_history[-1], D_A_history[-1]


def eri_ao2mo(C, g):
    pqrw = np.einsum("pqrs,sw->pqrw", g   , C)
    pqvw = np.einsum("pqrw,rv->pqvw", pqrw, C)
    puvw = np.einsum("pqvw,qu->puvw", pqvw, C)
    tuvw = np.einsum("puvw,pt->tuvw", puvw, C)
    return tuvw


def log_ci_states(ci_solver, thresh=1e-6):
    norb = ci_solver.norb
    nel_a, nel_b = ci_solver.nelec
    fci_occslst_a = fci.cistring.gen_occslst(range(norb), nel_a)
    fci_occslst_b = fci.cistring.gen_occslst(range(norb), nel_b)
    fci_strs_a = fci.cistring.make_strings(range(norb), nel_a)
    fci_strs_b = fci.cistring.make_strings(range(norb), nel_b)
    fci_space = len(fci_occslst_a) * len(fci_occslst_b)

    for root in range(ci_solver.nroots):
        if ci_solver.nroots == 1:
            ci_vector = ci_solver.ci
        else:
            ci_vector = ci_solver.ci[root]

        print(f"Logging CI vectors and coefficients > {thresh} for root number {root}:")

        # decimate the occs lists to get only the selected determinants
        if isinstance(ci_solver, fci.SCI):
            strs_a = np.intersect1d(ci_vector._strs[0], fci_strs_a, assume_unique=True)
            strs_b = np.intersect1d(ci_vector._strs[1], fci_strs_b, assume_unique=True)
            idx_a = fci.cistring.strs2addr(norb, nel_a, strs_a)
            idx_b = fci.cistring.strs2addr(norb, nel_b, strs_b)
            occslst_a = fci_occslst_a[idx_a]
            occslst_b = fci_occslst_b[idx_b]
            sci_space = len(occslst_a) * len(occslst_b)
            print(f"The SCI space contains {sci_space} determinants over {fci_space} ({sci_space/fci_space:.2%})")
        else:
            occslst_a = fci_occslst_a
            occslst_b = fci_occslst_b
            print(f"The FCI space contains {fci_space} determinants.")

        pad = 4 + norb
        coeffs_shown = 0
        print(f'  {"Conf": <{pad}} CI coefficients')
        for i, occsa in enumerate(occslst_a):
            for j, occsb in enumerate(occslst_b):
                if abs(ci_vector[i, j]) < thresh:
                    continue
                # generate the CI string and log it
                occ = ""
                for k in range(norb):
                    if k in occsa and k in occsb:
                        occ += "2"
                    elif k in occsa and k not in occsb:
                        occ += "u"
                    elif k not in occsa and k in occsb:
                        occ += "d"
                    else:
                        occ += "0"
                print("  %s     %+.8f" % (occ, ci_vector[i, j]))
                coeffs_shown += 1

        print(f"There are {coeffs_shown} CI coefficients > {thresh}\n")


def compute_sparsity(arr, threshold=1e-12):
    """
    Computes the sparsity of an N-dimensional NumPy array.

    Parameters:
    - arr: np.ndarray, input array
    - threshold: float, values below this threshold are considered as zeros

    Returns:
    - sparsity: float, ratio of near-zero elements
    """
    nonzero_elements = np.count_nonzero(np.abs(arr) > threshold)
    sparsity = nonzero_elements / arr.size
    return sparsity*100


def filter_ci_wf(ci_vector, nelec, norb, thresh=5e-3):
    if isinstance(nelec, tuple):
        nel_a, nel_b = nelec
    else:
        nel_a = nel_b = nelec//2
    fci_strs_a = fci.cistring.make_strings(range(norb), nel_a)
    fci_strs_b = fci.cistring.make_strings(range(norb), nel_b)

    strs_a = []
    strs_b = []
    idx_a = []
    idx_b = []
    for i, str_a in enumerate(fci_strs_a):
        for j, str_b in enumerate(fci_strs_b):
            # include strings with CI coefficients above threshold
            if abs(ci_vector[i, j]) >= thresh:
                idx_a.append(i)
                idx_b.append(j)
                strs_a.append(str_a)
                strs_b.append(str_b)

    idxs = np.intersect1d(idx_a, idx_b)
    strs = np.intersect1d(strs_a, strs_b)
    return ci_vector[idxs,:][:,idxs], (np.asarray(strs), np.asarray(strs))



if __name__ == '__main__':

    # oxygen at 2 Angstrom
    R = 2.0
    mol = gto.M(
        atom = f'N 0.0 0.0 0.0; \
                 N 0.0 0.0 {R};',
        basis='sto-3g',
        spin=0,
        verbose=3
    )

    # hydrogen chain
    nH = 8
    mol = gto.M()
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=3
    mol.atom = [['H', (0, 0, 6*i)] for i in range(nH)]
    mol.unit = 'B'
    mol.build()

    omega = 0.5
    mf = dft.RKS(mol)
    mf._numint.libxc = dft.xcfun
    mf.xc = f'ldaerf + lr_hf({omega})'
    mf.omega = omega

    # mf = scf.UHF(mol)
    # mf.kernel()

    mf.kernel()
    print(mf.mo_energy)
    # print(mf.nelec)

    # print orbitals
    from pyscf import tools
    tools.molden.dump_scf(mf, 'mf.molden')

    # active space
    active = [1,2,3,4,5,6,7,8]
    nmo_act = len(active)
    # active = (active, active)
    nel_act = 8

    # selected CI solver
    # for singlets
    # fci_solver = fci.selected_ci_spin0.SCI()

    # for higher multiplicities
    # fci_solver = fci.SCI()

    # fci_solver.max_cycle = 30
    # fci_solver.conv_tol = 1e-10
    # fci_solver.ci_coeff_cutoff = 5e-3
    # fci_solver.select_cutoff = 5e-3
    # print(f"SCI cutoffs: ci_coeff_cutoff = {fci_solver.fci_coeff_cutoff}, select_cutoff = {fci_solver.select_cutoff}")


    e,d = fci_srdft(mf, nel_act, nmo_act, active, max_iter=1, threshold=1e-8, debug=False, alpha=0.0)


    print('\nCorrelation energy:', e - mf.e_tot)
    print('\nAlpha active density matrix:\n', np.array2string(d[0], prefix=' '))
    print('\nBeta active density matrix:\n', np.array2string(d[1], prefix=' '))
