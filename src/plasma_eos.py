"""Load the generated Saha-Boltzmann Ar/H table for the jet simulations."""

from pathlib import Path

import numpy as np

from .constants import EV_TO_K

with np.load(Path(__file__).with_name("data") / "plasma_eos.npz") as data:
    N_GRID = data["n_m3"]
    T_GRID = data["T_eV"] * EV_TO_K
    # Adjacent density samples share cache lines at each temperature probe.
    E_TABLE = data["Ar_e_Jkg"].T.ravel()
    Z_TABLE = data["Ar_Z"].T.ravel()
    H_E_TABLE = data["H_e_Jkg"].T.ravel()
    H_Z_TABLE = data["H_Z"].T.ravel()
N_COUNT, T_COUNT = N_GRID.size, T_GRID.size
# density_cell specializes the generator's geometric grid and two extra knots.
assert np.allclose(N_GRID, np.unique(np.r_[np.geomspace(1e14, 1e28, 561), 2e22, 4.5e22]), rtol=1e-14, atol=0)
assert all(table.size == N_COUNT * T_COUNT for table in (E_TABLE, Z_TABLE, H_E_TABLE, H_Z_TABLE))
assert all(np.all(np.diff(table.reshape(T_COUNT, N_COUNT), axis=0) > 0) for table in (E_TABLE, H_E_TABLE))
EOS_CONSTANTS = {"eos_n": N_GRID, "eos_T": T_GRID, "eos_e": E_TABLE, "eos_z": Z_TABLE, "eos_h_e": H_E_TABLE, "eos_h_z": H_Z_TABLE}
