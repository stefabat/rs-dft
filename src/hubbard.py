import numpy as np
import pandas as pd
from pyscf import gto, scf, ao2mo, fci
from typing import Tuple, Dict
from opt_einsum import contract
np.set_printoptions(precision=8, suppress=True, linewidth=200)


def generate_hopping_matrix(
    lattice_type: str,
    size: Tuple[int, ...],
    t: float = 1.0,
    periodic: bool = True
) -> np.ndarray:
    """
    Generate hopping matrix for different lattice types.

    Parameters:
    -----------
    lattice_type : str
        Type of lattice ('chain', 'square', 'triangular', 'honeycomb')
    size : tuple
        Dimensions of the lattice. For 1D: (N,), for 2D: (Nx, Ny)
    t : float, optional
        Hopping parameter strength (default: 1.0)
    periodic : bool, optional
        Whether to use periodic boundary conditions (default: True)

    Returns:
    --------
    np.ndarray
        Hopping matrix for the specified lattice
    """

    if lattice_type == 'chain':
        if len(size) != 1:
            raise ValueError("Chain lattice requires 1D size tuple (N,)")
        N = size[0]
        matrix = np.zeros((N, N))

        # Nearest neighbor hopping
        for i in range(N-1):
            matrix[i, i+1] = matrix[i+1, i] = -t

        # Periodic boundary conditions
        if periodic:
            matrix[0, N-1] = matrix[N-1, 0] = -t

        return matrix

    elif lattice_type == 'square':
        if len(size) != 2:
            raise ValueError("Square lattice requires 2D size tuple (Nx, Ny)")
        Nx, Ny = size
        N = Nx * Ny
        matrix = np.zeros((N, N))

        for i in range(N):
            x, y = i // Ny, i % Ny

            # Right neighbor
            if x < Nx-1:
                j = i + Ny
                matrix[i, j] = matrix[j, i] = -t

            # Up neighbor
            if y < Ny-1:
                j = i + 1
                matrix[i, j] = matrix[j, i] = -t

            # Periodic boundary conditions
            if periodic:
                # Right edge to left edge
                if x == Nx-1:
                    j = i - (Nx-1)*Ny
                    matrix[i, j] = matrix[j, i] = -t

                # Top edge to bottom edge
                if y == Ny-1:
                    j = i - (Ny-1)
                    matrix[i, j] = matrix[j, i] = -t

        return matrix

    elif lattice_type == 'triangular':
        if len(size) != 2:
            raise ValueError("Triangular lattice requires 2D size tuple (Nx, Ny)")
        Nx, Ny = size
        N = Nx * Ny
        matrix = np.zeros((N, N))

        for i in range(N):
            x, y = i // Ny, i % Ny

            # Same as square lattice
            # Right neighbor
            if x < Nx-1:
                j = i + Ny
                matrix[i, j] = matrix[j, i] = -t

            # Up neighbor
            if y < Ny-1:
                j = i + 1
                matrix[i, j] = matrix[j, i] = -t

            # Additional diagonal neighbor
            if x < Nx-1 and y < Ny-1:
                j = i + Ny + 1
                matrix[i, j] = matrix[j, i] = -t

            # Periodic boundary conditions
            if periodic:
                # Right edge to left edge
                if x == Nx-1:
                    j = i - (Nx-1)*Ny
                    matrix[i, j] = matrix[j, i] = -t

                # Top edge to bottom edge
                if y == Ny-1:
                    j = i - (Ny-1)
                    matrix[i, j] = matrix[j, i] = -t

                # Diagonal wrap-around
                if x == Nx-1 and y == Ny-1:
                    j = i - (Nx-1)*Ny - (Ny-1)
                    matrix[i, j] = matrix[j, i] = -t

        return matrix

    elif lattice_type == 'honeycomb':
        if len(size) != 2:
            raise ValueError("Honeycomb lattice requires 2D size tuple (Nx, Ny)")
        Nx, Ny = size
        N = Nx * Ny * 2  # 2 sites per unit cell
        matrix = np.zeros((N, N))

        for i in range(0, N, 2):
            x, y = (i//2) // Ny, (i//2) % Ny

            # Connect A to B within unit cell
            matrix[i, i+1] = matrix[i+1, i] = -t

            # Connect to neighboring unit cells
            if x < Nx-1:  # Right neighbor
                j = i + 2*Ny
                matrix[i+1, j] = matrix[j, i+1] = -t

            if y < Ny-1:  # Upper neighbor
                j = i + 2
                matrix[i+1, j] = matrix[j, i+1] = -t

            # Periodic boundary conditions
            if periodic:
                if x == Nx-1:  # Right edge to left edge
                    j = i - 2*(Nx-1)*Ny
                    matrix[i+1, j] = matrix[j, i+1] = -t

                if y == Ny-1:  # Top edge to bottom edge
                    j = i - 2*(Ny-1)
                    matrix[i+1, j] = matrix[j, i+1] = -t

        return matrix

    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")


def compute_double_occupancy(rdm1_up: np.ndarray, rdm1_dn: np.ndarray,
                           rdm2: np.ndarray) -> float:
    """
    Compute average double occupancy <ni↑ni↓>

    Parameters:
    -----------
    rdm1_up : np.ndarray
        Spin-up one-particle reduced density matrix
    rdm1_dn : np.ndarray
        Spin-down one-particle reduced density matrix
    rdm2 : np.ndarray
        Two-particle reduced density matrix

    Returns:
    --------
    float
        Average double occupancy
    """
    n_sites = len(rdm1_up)
    double_occ = 0.0

    # Sum over diagonal elements for each site
    for i in range(n_sites):
        double_occ += rdm2[i,i,i,i]

    return double_occ / n_sites


def compute_local_moment(rdm1_up: np.ndarray, rdm1_dn: np.ndarray) -> np.ndarray:
    """
    Compute local magnetic moment <(ni↑ - ni↓)²> for each site

    Parameters:
    -----------
    rdm1_up : np.ndarray
        Spin-up one-particle reduced density matrix
    rdm1_dn : np.ndarray
        Spin-down one-particle reduced density matrix

    Returns:
    --------
    np.ndarray
        Local magnetic moment for each site
    """
    n_sites = len(rdm1_up)
    local_moments = np.zeros(n_sites)

    for i in range(n_sites):
        n_up = rdm1_up[i,i].real
        n_dn = rdm1_dn[i,i].real
        local_moments[i] = (n_up - n_dn)**2

    return local_moments


def compute_spin_correlations(rdm1_up: np.ndarray, rdm1_dn: np.ndarray,
                            rdm2: np.ndarray) -> np.ndarray:
    """
    Compute spin-spin correlation function <Si·Sj>

    Parameters:
    -----------
    rdm1_up : np.ndarray
        Spin-up one-particle reduced density matrix
    rdm1_dn : np.ndarray
        Spin-down one-particle reduced density matrix
    rdm2 : np.ndarray
        Two-particle reduced density matrix

    Returns:
    --------
    np.ndarray
        Matrix of spin-spin correlations between all pairs of sites
    """
    n_sites = len(rdm1_up)
    spin_corr = np.zeros((n_sites, n_sites))

    for i in range(n_sites):
        for j in range(n_sites):
            # <Sz_i Sz_j>
            sz_corr = 0.25 * ((rdm1_up[i,i] - rdm1_dn[i,i]) *
                             (rdm1_up[j,j] - rdm1_dn[j,j]))

            # <S+_i S-_j> = <c†_i↑ c_i↓ c†_j↓ c_j↑>
            splus_sminus = rdm2[i,j,j,i]

            # Total correlation = <Sz_i Sz_j> + 1/2(<S+_i S-_j> + <S-_i S+_j>)
            spin_corr[i,j] = sz_corr.real + 0.5 * (splus_sminus + splus_sminus.conj()).real

    return spin_corr


def compute_staggered_magnetization(rdm1_up: np.ndarray,
                                  rdm1_dn: np.ndarray,
                                  lattice_type: str = 'chain') -> float:
    """
    Compute staggered magnetization (AFM order parameter)

    Parameters:
    -----------
    rdm1_up : np.ndarray
        Spin-up one-particle reduced density matrix
    rdm1_dn : np.ndarray
        Spin-down one-particle reduced density matrix
    lattice_type : str
        Type of lattice ('chain' or 'square')

    Returns:
    --------
    float
        Staggered magnetization
    """
    n_sites = len(rdm1_up)
    local_sz = np.zeros(n_sites)

    # Compute local Sz for each site
    for i in range(n_sites):
        local_sz[i] = 0.5 * (rdm1_up[i,i] - rdm1_dn[i,i]).real

    if lattice_type == 'chain':
        # For 1D chain, alternate signs
        stag_factors = np.array([-1.0**i for i in range(n_sites)])
    elif lattice_type == 'square':
        # For 2D square lattice, checkerboard pattern
        n = int(np.sqrt(n_sites))
        stag_factors = np.array([-1.0**(i+j) for i in range(n) for j in range(n)])
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")

    m_stag = np.abs(np.mean(local_sz * stag_factors))
    return m_stag


def compute_pair_correlations(rdm2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute singlet and triplet pairing correlations

    Parameters:
    -----------
    rdm2 : np.ndarray
        Two-particle reduced density matrix

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Singlet and triplet pairing correlation matrices
    """
    n_sites = int(np.sqrt(rdm2.shape[0]))
    singlet_corr = np.zeros((n_sites, n_sites), dtype=complex)
    triplet_corr = np.zeros((n_sites, n_sites), dtype=complex)

    for i in range(n_sites):
        for j in range(n_sites):
            # Singlet: <c†_i↑c†_i↓c_j↓c_j↑>
            singlet_corr[i,j] = rdm2[i,i,j,j]

            # Triplet: <c†_i↑c†_j↓c_i↓c_j↑>
            triplet_corr[i,j] = rdm2[i,j,i,j]

    return singlet_corr, triplet_corr


def transform_1rdm(rdm1_mo: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Transform 1-RDM from MO to AO basis using einsum.
    For real orbitals: D_AO = C @ D_MO @ C^T

    Parameters:
    -----------
    rdm1_mo : np.ndarray
        One-particle RDM in MO basis
    C : np.ndarray
        MO coefficients matrix (AO to MO transformation)

    Returns:
    --------
    np.ndarray
        One-particle RDM in AO basis
    """
    return contract('µp,pq,νq->µν', C, rdm1_mo, C)


def transform_2rdm(rdm2_mo: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Transform 2-RDM from MO to AO basis using einsum.
    For real orbitals: D_AO(µνλσ) = C_µp C_νq C_λr C_σs D_MO(pqrs)

    Parameters:
    -----------
    rdm2_mo : np.ndarray
        Two-particle RDM in MO basis
    C : np.ndarray
        MO coefficients matrix (AO to MO transformation)

    Returns:
    --------
    np.ndarray
        Two-particle RDM in AO basis
    """
    return contract('µp,νq,λr,σs,pqrs->µνλσ', C, C, C, C, rdm2_mo)


def transform_spinrdm_mo_to_ao(rdm1_up_mo: np.ndarray,
                               rdm1_dn_mo: np.ndarray,
                               rdm2_mo: np.ndarray,
                               C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform spin-resolved RDMs from MO to AO basis.
    Optimized for real orbitals.

    Parameters:
    -----------
    rdm1_up_mo : np.ndarray
        Spin-up 1-RDM in MO basis
    rdm1_dn_mo : np.ndarray
        Spin-down 1-RDM in MO basis
    rdm2_mo : np.ndarray
        2-RDM in MO basis
    C : np.ndarray
        MO coefficients matrix

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Spin-up 1-RDM, spin-down 1-RDM, and 2-RDM in AO basis
    """
    rdm1_up_ao = transform_1rdm(rdm1_up_mo, C)
    rdm1_dn_ao = transform_1rdm(rdm1_dn_mo, C)
    rdm2_ao = transform_2rdm(rdm2_mo, C)

    return rdm1_up_ao, rdm1_dn_ao, rdm2_ao


def init_df(U_values: np.ndarray) -> pd.DataFrame:
    """
    Create empty DataFrame for storing Hubbard model observables.

    Parameters:
    -----------
    U_values : np.ndarray
        Array of U/t values

    Returns:
    --------
    pd.DataFrame
        DataFrame with U values as index and observable columns
    """
    columns = [
        'U_over_t',          # Coupling strength
        'E_tot',             # Total energy
        'E_per_site',        # Energy per site
        'E_corr',            # Correlation energy (compared to HF)
        'double_occ_avg',    # Average double occupancy
        'local_moment_avg',  # Average local moment <(n↑-n↓)²>
        'sz_avg',            # Average Sz
        'sz_staggered',      # Staggered magnetization
        'nn_spin_corr',      # Nearest-neighbor spin correlation
        'nnn_spin_corr',     # Next-nearest-neighbor spin correlation
    ]

    # Create DataFrame with NaN values
    df = pd.DataFrame(index=U_values, columns=columns)
    df['U_over_t'] = U_values  # Fill U/t values
    return df


def compute_observables(
    U_value: float,
    N_sites: int,
    rdm1_up_site: np.ndarray,
    rdm1_dn_site: np.ndarray,
    rdm2_site: np.ndarray,
    E_tot: float,
    E_hf: float = None
) -> Dict[str, float]:
    """
    Compute all observables from RDMs and return as dictionary.

    Parameters:
    -----------
    U_value : float
        Current U/t value
    rdm1_up_site, rdm1_dn_site : np.ndarray
        1-RDMs in site basis for up and down spins
    rdm2_site : np.ndarray
        2-RDM in site basis
    E_tot : float
        Total energy
    E_hf : float, optional
        HF energy for correlation energy computation
    N_sites : int
        Number of sites

    Returns:
    --------
    Dict[str, float]
        Dictionary of computed observables
    """
    # Initialize results dictionary
    results = {
        'U_over_t': U_value,
        'E_tot': E_tot,
        'E_per_site': E_tot / N_sites,
        'E_corr': E_tot - E_hf if E_hf is not None else np.nan
    }

    # Compute double occupancy
    double_occ = 0.0
    for i in range(N_sites):
        double_occ += rdm2_site[i,i,i,i]
    results['double_occ_avg'] = double_occ / N_sites

    # Local magnetic moment
    local_moment = 0.0
    for i in range(N_sites):
        n_up = rdm1_up_site[i,i]
        n_dn = rdm1_dn_site[i,i]
        local_moment += (n_up - n_dn)**2
    results['local_moment_avg'] = local_moment / N_sites

    # Magnetization and staggered magnetization
    sz_values = np.array([0.5 * (rdm1_up_site[i,i] - rdm1_dn_site[i,i])
                         for i in range(N_sites)])
    results['sz_avg'] = np.mean(sz_values)
    results['sz_staggered'] = np.mean(sz_values * np.array([-1.0]**i
                                                          for i in range(N_sites)))

    # Spin correlations
    spin_corr = compute_spin_correlations(rdm1_up_site, rdm1_dn_site, rdm2_site)
    # Nearest neighbor
    nn_corr = 0.0
    nnn_corr = 0.0
    for i in range(N_sites):
        nn_corr += spin_corr[i, (i+1)%N_sites]  # Periodic boundary
        nnn_corr += spin_corr[i, (i+2)%N_sites]  # Next-nearest neighbor

    results['nn_spin_corr'] = nn_corr / N_sites
    results['nnn_spin_corr'] = nnn_corr / N_sites

    return results

def update_df(df: pd.DataFrame, U_value: float,
                         observables: Dict[str, float]) -> pd.DataFrame:
    """
    Update DataFrame with computed observables at given U value.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to update
    U_value : float
        Current U/t value
    observables : Dict[str, float]
        Dictionary of computed observables

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame
    """
    df.loc[U_value] = pd.Series(observables)
    return df


if __name__ == "__main__":

    # at half-filling, we have as many electrons as sites
    N = 8
    h1 = generate_hopping_matrix('chain', (N,), t=-1.0, periodic=True)

    # Generate data (placeholder for your actual calculations)
    U_values = np.linspace(0, 1, 1)
    print(U_values)

    df_hf  = init_df(U_values)
    df_sci = init_df(U_values)
    df_fci = init_df(U_values)

    for U in U_values:
        print(f'Running for U = {U}')

        mol = gto.M()
        mol.nelectron = N
        mol.verbose = 4
        mol.incore_anyway = True

        eri = np.zeros((N,N,N,N))
        for i in range(N):
            eri[i,i,i,i] = U

        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(N)
        mf._eri = ao2mo.restore(8, eri, N)
        mf.init_guess = '1e'

        # this should be mean-field Hubbard
        mf.kernel()
        E_hf = mf.e_tot
        C = mf.mo_coeff
        C_occ = C[:,mf.mo_occ > 0]

        D_ao = mf.make_rdm1()

        print(transform_1rdm(D_ao, C.T))
        d = mf.make_rdm2()
        print(transform_2rdm(d,C.T))

        # D_ao = transform_1rdm(D, C)
        # d_ao = transform_2rdm(d, C)


        # obs_hf = compute_and_store_observables(U, rdm1_up_hf, rdm1_dn_hf, rdm2_hf,
        #                                     E_hf)
        # df_hf = update_observables_df(df_hf, U, obs_hf)

        # mf.CISD().run()

        # mf.CCSD().run()

        # fci_solver = fci.FCI(mf)
        # efci, vfci = fci_solver.kernel()

        # D = fci_solver.make_rdm1(vfci, N, N)
        # print(D)
        # C = mf.mo_coeff
        # print(np.dot(C, np.dot(D, C.T)))

        # print(f'E(FCI) = {efci}  E_corr = {efci - mf.e_tot}')
