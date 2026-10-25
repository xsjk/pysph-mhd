"""Single-species plasma EOS lookup shared by Python and CUDA."""

from math import log, sqrt

import numpy as np
from compyle.low_level import cast
from compyle.types import annotate, declare


def __assert_fail(message, _file, _line, _function):
    # Python equivalent of CUDA's intrinsic; do not transpile this function.
    raise AssertionError(message)


@annotate(value="double", grid="gdoublep", count="int", return_="int")
def density_cell(value, grid, count):
    # Lower cell index on the log grid with two extra density knots.
    row = declare("int")
    # The base grid has 40 intervals per decade, starting at 1e14 m^-3.
    row = cast(log(value / 1e14) * 17.37177927613007, "int")
    if value >= 2e22:
        row += 1
    if value >= 4.5e22:
        row += 1
    row = min(row, count - 2)
    # Correct rounding at knots, including the two extra initial-state knots.
    if value < grid[row]:
        row -= 1
    elif row < count - 2 and value >= grid[row + 1]:
        row += 1
    return row


@annotate(table="gdoublep", row="int", column="int", fraction="double", width="int", return_="double")
def density_value(table, row, column, fraction, width):
    offset = declare("int")
    offset = row * width + column
    return table[offset] + fraction * (table[offset + width] - table[offset])


@annotate(rho="double", energy="double", n_grid="gdoublep", t_grid="gdoublep", e_table="gdoublep", z_table="gdoublep", state="doublep", n_count="int", t_count="int", nucleus_mass="double", k_b="double")
def species_state(rho, energy, n_grid, t_grid, e_table, z_table, state, n_count, t_count, nucleus_mass, k_b):
    # Write [T, mean charge, pressure, sound speed].
    # Adiabatic sound speed follows de = p/rho**2 drho at ds=0.
    row, low, high, middle, column, offset = declare("int", 6)
    n = rho / nucleus_mass
    if not (n >= n_grid[0] and n <= n_grid[n_count - 1]):
        __assert_fail("Number density outside EOS table", "tabulated_eos.py", 0, "species_state")
    row = density_cell(n, n_grid, n_count)
    dn = n_grid[row + 1] - n_grid[row]
    fn = (n - n_grid[row]) / dn
    emin = density_value(e_table, row, 0, fn, t_count)
    emax = density_value(e_table, row, t_count - 1, fn, t_count)
    if not (energy >= emin and energy <= emax):
        __assert_fail("Energy outside EOS table", "tabulated_eos.py", 0, "species_state")
    low = 0
    high = t_count - 1
    while high - low > 1:
        middle = cast(0.5 * (low + high), "int")
        if energy < density_value(e_table, row, middle, fn, t_count):
            high = middle
        else:
            low = middle
    column = low
    e1 = density_value(e_table, row, column, fn, t_count)
    e2 = density_value(e_table, row, column + 1, fn, t_count)
    ft = (energy - e1) / (e2 - e1)
    dt = t_grid[column + 1] - t_grid[column]
    temperature = t_grid[column] + ft * dt
    z1 = density_value(z_table, row, column, fn, t_count)
    z2 = density_value(z_table, row, column + 1, fn, t_count)
    charge = z1 + ft * (z2 - z1)
    offset = row * t_count + column
    en1 = (e_table[offset + t_count] - e_table[offset]) / dn
    en2 = (e_table[offset + t_count + 1] - e_table[offset + 1]) / dn
    zn1 = (z_table[offset + t_count] - z_table[offset]) / dn
    zn2 = (z_table[offset + t_count + 1] - z_table[offset + 1]) / dn
    e_rho = (en1 + ft * (en2 - en1)) / nucleus_mass
    e_temperature = (e2 - e1) / dt
    z_n = zn1 + ft * (zn2 - zn1)
    z_temperature = (z2 - z1) / dt
    pressure = (1.0 + charge) * n * k_b * temperature
    p_rho = k_b * temperature * (1.0 + charge + n * z_n) / nucleus_mass
    p_temperature = n * k_b * (1.0 + charge + temperature * z_temperature)
    sound_squared = p_rho + p_temperature * (pressure / (rho * rho) - e_rho) / e_temperature
    if not (sound_squared > 0.0):
        __assert_fail("Nonpositive or invalid sound speed squared", "tabulated_eos.py", 0, "species_state")
    state[0] = temperature
    state[1] = charge
    state[2] = pressure
    state[3] = sqrt(sound_squared)


EOS_HELPERS = [density_cell, density_value, species_state]


def tabulated_energy_at_temperature(number_density, temperature, n_grid, t_grid, e_table):
    # Energy in J/kg, using the same bilinear interpolation as CUDA.
    n, t = np.broadcast_arrays(number_density, temperature)
    assert np.all((n >= n_grid[0]) & (n <= n_grid[-1]))
    assert np.all((t >= t_grid[0]) & (t <= t_grid[-1]))
    row = np.minimum(np.searchsorted(n_grid, n, side="right") - 1, n_grid.size - 2)
    column = np.minimum(np.searchsorted(t_grid, t, side="right") - 1, t_grid.size - 2)
    fn = (n - n_grid[row]) / (n_grid[row + 1] - n_grid[row])
    ft = (t - t_grid[column]) / (t_grid[column + 1] - t_grid[column])
    e1 = density_value(e_table, row, column, fn, t_grid.size)
    e2 = density_value(e_table, row, column + 1, fn, t_grid.size)
    return e1 + ft * (e2 - e1)
