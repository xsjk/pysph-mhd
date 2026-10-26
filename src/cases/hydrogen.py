"""Hydrogen plasmoids launched from the configured latitude rings."""

from dataclasses import dataclass
from math import isfinite, pi
from typing import override

import numpy as np
from pysph.base.utils import get_particle_array
from scipy.optimize import minimize_scalar
from scipy.special import j0, j1, jn_zeros

from ..config import SimulationConfig
from ..constants import EV_TO_K, H_MASS, K_B
from ..tabulated_eos import species_state
from .jets import JetApplication, LatitudeRingsConfig, jet_basis, latitude_rings
from .lattice import close_packed_lattice

HYDROGEN_MATERIAL_ID = 2
CYLINDRICAL_ROOT = float(jn_zeros(1, 1)[0])


@dataclass(frozen=True)
class HydrogenGunConfig:
    radius: float
    length: float
    axial_speed: float
    temperature_ev: float
    number_density: float
    peak_field: float


@dataclass(frozen=True)
class HydrogenConfig(SimulationConfig):
    jet: HydrogenGunConfig
    layout: LatitudeRingsConfig

    @override
    def validate(self):
        super().validate()
        for value in vars(self.jet).values():
            assert type(value) is float
            assert isfinite(value)
            assert value > 0.0
        self.layout.validate()
        assert self.numerics.periodic_mode == "none"


def field_parameters(gun):
    kr = CYLINDRICAL_ROOT / gun.radius
    kz = pi / gun.length
    lam = np.hypot(kr, kz)

    # |B| is largest at the axial midplane. Optimize only the radial coordinate,
    # including both endpoints, so normalization never depends on the lattice.
    def magnitude_squared(q):
        return lam**2 * j1(q) ** 2 + kr**2 * j0(q) ** 2

    optimum = minimize_scalar(lambda q: -magnitude_squared(q), bounds=(0.0, CYLINDRICAL_ROOT), method="bounded", options={"xatol": 1.0e-13})
    assert optimum.success
    maximum = max(magnitude_squared(0.0), magnitude_squared(CYLINDRICAL_ROOT), -optimum.fun)
    return kr, kz, lam, gun.peak_field / np.sqrt(maximum)


def cylindrical_spheromak_field(x, y, axial, gun):
    """Evaluate the interior field in right-handed local (x, y, axial).

    Returns
    -------
    tuple
        Cartesian field components in tesla, with rear=0 and front=L.
    """
    assert x.shape == y.shape == axial.shape
    kr, kz, lam, amplitude = field_parameters(gun)
    radius = np.hypot(x, y)
    inverse_radius = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
    radial = -amplitude * kz * j1(kr * radius) * np.cos(kz * axial)
    azimuthal = amplitude * lam * j1(kr * radius) * np.sin(kz * axial)
    # The reference MATLAB uses kz here; kr is required for div(B)=0 and curl(B)=lambda*B.
    longitudinal = amplitude * kr * j0(kr * radius) * np.sin(kz * axial)
    return (
        (radial * x - azimuthal * y) * inverse_radius,
        (radial * y + azimuthal * x) * inverse_radius,
        longitudinal,
    )


class HydrogenJets(JetApplication):
    config_type = HydrogenConfig

    @property
    @override
    def bounds(self):
        extent = self.config.layout.radius + self.config.jet.length
        return (-extent, extent) * 3

    @override
    def create_particles(self):
        (particle,) = super().create_particles()
        # This also initializes the held, nonintegrated H array during Ar preflight.
        if self.config.physics.plasma_eos:
            from ..plasma_eos import H_E_TABLE, H_Z_TABLE, N_COUNT, N_GRID, T_COUNT, T_GRID  # ruff: ignore[import-outside-top-level] -- Only enabled plasma runs need the table.

            names = ("T", "mean_charge", "p", "cs")
            state = np.empty(4)
            species_state(particle.rho[0], particle.e[0], N_GRID, T_GRID, H_E_TABLE, H_Z_TABLE, state, N_COUNT, T_COUNT, H_MASS, K_B, -1)
        else:
            names = ("p", "cs")
            pressure = (self.gamma - 1) * particle.rho[0] * particle.e[0]
            state = (pressure, np.sqrt(self.gamma * pressure / particle.rho[0]))
        for name, value in zip(names, state, strict=True):
            getattr(particle, name)[:] = value
        if particle.gpu is not None:
            particle.gpu.push(*names)
        return [particle]

    @override
    def refresh_initial_thermodynamics(self, particle):
        particle.e[:] = self.energy_at_temperature(particle.rho / H_MASS, self.config.jet.temperature_ev * EV_TO_K, HYDROGEN_MATERIAL_ID)

    @override
    def create_mhd_particles(self):
        gun = self.config.jet
        spacing = 2.0 * gun.radius / self.nx
        axial, cross_x, cross_y = close_packed_lattice((0.0, gun.length, -gun.radius, gun.radius, -gun.radius, gun.radius), spacing)
        inside = (cross_x**2 + cross_y**2 < gun.radius**2) & (axial <= gun.length)
        axial, cross_x, cross_y = axial[inside], cross_x[inside], cross_y[inside]
        local_position = np.vstack((cross_x, cross_y, axial - gun.length))
        local_field = np.asarray(cylindrical_spheromak_field(cross_x, cross_y, axial, gun))
        positions, fields, velocities = [], [], []
        fronts, directions = latitude_rings(self.config.layout)
        for front, direction in zip(fronts, directions, strict=True):
            basis = jet_basis(direction)
            positions.append(basis @ local_position + front[:, None])
            fields.append(basis @ local_field)
            velocities.append(np.tile(basis[:, 2, None] * gun.axial_speed, (1, len(axial))))
        x, y, z = np.concatenate(positions, axis=1)
        bx, by, bz = np.concatenate(fields, axis=1)
        u, v, w = np.concatenate(velocities, axis=1)
        count = len(x)
        rho = gun.number_density * H_MASS
        energy = float(self.energy_at_temperature(gun.number_density, gun.temperature_ev * EV_TO_K, HYDROGEN_MATERIAL_ID))
        volume = pi * gun.radius**2 * gun.length / len(axial)
        particle = get_particle_array(
            name="fluid",
            x=x,
            y=y,
            z=z,
            u=u,
            v=v,
            w=w,
            rho=np.full(count, rho),
            e=np.full(count, energy),
            h=np.full(count, self.hfact * volume ** (1.0 / 3.0)),
            m=np.full(count, rho * volume),
            gid=np.arange(count, dtype=np.uint32),
            Bx=bx,
            By=by,
            Bz=bz,
            alpha1=np.zeros(count),
        )
        particle.add_property("material_id", type="int", data=np.full(count, HYDROGEN_MATERIAL_ID, dtype=np.int32))
        particle.add_property("source_id", type="int", data=np.repeat(np.arange(1, len(fronts) + 1, dtype=np.int32), len(axial)))
        return [particle]
