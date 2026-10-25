"""Generate the Ar/H Saha-Boltzmann energy and mean-charge tables."""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.constants import c, electron_mass, elementary_charge, h
from scipy.optimize import brentq
from scipy.special import logsumexp

from .constants import ARGON_MASS, H_MASS

# Ar ionization thresholds: NIST Handbook (Ar I) and Saloman (2010),
# https://srd.nist.gov/jpcrdreprint/1.3337661.pdf (Ar II-XVIII).
ARGON_IONIZATION_EV = np.array((15.759610, 27.62995, 40.735, 59.58, 74.84, 91.29, 124.41, 143.458, 422.60, 479.84, 540.4, 619.0, 685.47, 755.13, 855.47, 918.33, 4120.6653, 4426.2225))
ARGON_GROUND_WEIGHTS = (1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1)


def atomic_species(argon_source):
    # Extract Ar I-VI bound levels without executing the legacy source file.
    stages = ("I", "II", "III", "IV", "V", "VI")
    names = {prefix + stage for stage in stages for prefix in ("E_Ar", "g_Ar")}
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.walk(ast.parse(argon_source.read_text())) if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}
    assert assignments.keys() == names
    argon = [(np.array([0.0]), np.array([weight])) for weight in ARGON_GROUND_WEIGHTS]
    for charge, stage in enumerate(stages):
        energy = np.asarray(assignments["E_Ar" + stage], dtype=float)
        weights = np.asarray(assignments["g_Ar" + stage], dtype=float)
        assert energy.shape == weights.shape
        assert np.all(energy >= 0)
        assert np.all(weights > 0)
        assert energy[0] == 0
        assert weights[0] == ARGON_GROUND_WEIGHTS[charge]
        bound = energy < ARGON_IONIZATION_EV[charge]
        argon[charge] = energy[bound], weights[bound]

    # H fine-structure levels through n=5 and ionization limit, in cm^-1:
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm
    hydrogen_cm = np.array((0.0, 82258.9191, 82259.2850, 82258.9544, 97492.2112, 97492.3196, 97492.2217, 97492.3195, 97492.3556, 102823.8486, 102823.8943, 102823.8530, 102823.8942, 102823.9095, 102823.9095, 102823.9171, 105291.6287, 105291.6521, 105291.6309, 105291.6520, 105291.6599, 105291.6598, 105291.6637, 105291.6637, 105291.6661))
    hydrogen_weights = np.array((2, 2, 4, 2, 2, 4, 2, 4, 6, 2, 4, 2, 4, 6, 6, 8, 2, 4, 2, 4, 6, 6, 8, 8, 10))
    cm_to_ev = 100 * h * c / elementary_charge
    hydrogen = [(hydrogen_cm * cm_to_ev, hydrogen_weights), (np.array([0.0]), np.array([1.0]))]
    return {"Ar": (ARGON_MASS, ARGON_IONIZATION_EV, argon), "H": (H_MASS, np.array([109678.7717 * cm_to_ev]), hydrogen)}


def partition_moments(temperature_ev, levels):
    # Partition function and mean excitation energy for each charge state.
    moments = []
    for energies, weights in levels:
        boltzmann = weights * np.exp(-energies / temperature_ev)
        partition = float(boltzmann.sum())
        excitation = float(np.dot(boltzmann / partition, energies))
        moments.append((partition, excitation))
    return np.asarray(moments).T


def table_column(temperature, n_grid, mass, ionization, levels):
    # Compute temperature-dependent Saha weights once for the whole density column.
    partition_functions, excitation_ev = partition_moments(temperature, levels)
    charges = np.arange(ionization.size + 1)
    log_charges = np.log(charges[1:])
    binding_ev = np.r_[0.0, np.cumsum(ionization)]
    log_lambda_e = 0.5 * np.log(h**2 / (2 * np.pi * electron_mass * elementary_charge * temperature))
    log_prefactor = np.log(2.0) - 3 * log_lambda_e
    fixed_weights = np.log(partition_functions / partition_functions[0]) + charges * log_prefactor - binding_ev / temperature

    def neutrality(log_ne, log_n):
        log_weights = fixed_weights - charges * log_ne
        log_charge = logsumexp(log_weights[1:] + log_charges) - logsumexp(log_weights)
        return log_ne - log_n - log_charge

    states = []
    for number_density in n_grid:
        log_n = np.log(number_density)
        lower = min(log_n - 80, 0.5 * (log_n + log_prefactor + np.log(partition_functions[1] / partition_functions[0]) - ionization[0] / temperature) - 80)
        log_ne = brentq(neutrality, lower, log_n + np.log(float(charges.size)), args=(log_n,), xtol=1e-11, rtol=1e-13)
        assert abs(neutrality(log_ne, log_n)) < 1e-8
        log_weights = fixed_weights - charges * log_ne
        fractions = np.exp(log_weights - logsumexp(log_weights))
        charge = float(np.dot(fractions, charges))
        bound = float(np.dot(fractions, binding_ev))
        excitation = float(np.dot(fractions, excitation_ev))
        energy = elementary_charge / mass * (1.5 * (1 + charge) * temperature + bound + excitation)
        states.append((energy, charge))
    return np.asarray(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--argon-levels", type=Path, required=True, help="Legacy spmhd plasmaProperties.py containing E_ArI-VI and g_ArI-VI")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    species = atomic_species(args.argon_levels)
    n_grid = np.unique(np.r_[np.geomspace(1e14, 1e28, 561), 2e22, 4.5e22])
    t_grid = np.unique(np.r_[np.geomspace(0.001, 1e5, 801), 1.5, 2.0])
    tables = {"n_m3": n_grid, "T_eV": t_grid}
    for name, (mass, ionization, levels) in species.items():
        states = np.empty((n_grid.size, t_grid.size, 2))
        generate_column = partial(table_column, n_grid=n_grid, mass=mass, ionization=ionization, levels=levels)
        with ProcessPoolExecutor(max_workers=4) as executor:
            for column, values in enumerate(executor.map(generate_column, t_grid)):
                states[:, column] = values
                if column % 100 == 0:
                    print(f"{name}: {column + 1}/{t_grid.size} temperature columns", flush=True)
        energy, charge = states[:, :, 0], states[:, :, 1]
        assert np.all(np.isfinite(states))
        assert np.all(np.diff(energy, axis=1) > 0)
        assert np.all((charge >= 0) & (charge <= ionization.size + 1e-11))
        tables[name + "_e_Jkg"], tables[name + "_Z"] = energy, charge
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plasma_eos.npz"
    np.savez_compressed(path, **tables)
    print(path, flush=True)


if __name__ == "__main__":
    main()
