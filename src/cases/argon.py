from dataclasses import dataclass
from math import pi
from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..config import SimulationConfig
from .jets import JetApplication, LatitudeRingsConfig, jet_basis, latitude_rings
from .lattice import close_packed_lattice

ARGON_MASS_NUMBER = 39.948
ARGON_MOLAR_MASS = 0.039948
ATOMIC_MASS = 1.6726231e-27
EV_TO_K = 11604.505
GAS_CONSTANT = 8.31

ARGON_MATERIAL_ID = 0


@dataclass(frozen=True)
class ArgonGunConfig:
    length: float
    radius: float
    end_taper_fraction: float
    axial_speed: float
    azimuthal_speed: float
    temperature_ev: float
    number_density: float


@dataclass(frozen=True)
class ArgonConfig(SimulationConfig):
    jet: ArgonGunConfig
    layout: LatitudeRingsConfig

    @override
    def validate(self):
        super().validate()
        assert type(self.jet.length) is float
        assert self.jet.length > 0.0
        assert type(self.jet.radius) is float
        assert self.jet.radius > 0.0
        assert type(self.jet.end_taper_fraction) is float
        assert 0.0 < self.jet.end_taper_fraction <= 0.5
        assert type(self.jet.axial_speed) is float
        assert self.jet.axial_speed > 0.0
        assert type(self.jet.azimuthal_speed) is float
        assert self.jet.azimuthal_speed >= 0.0
        assert type(self.jet.temperature_ev) is float
        assert self.jet.temperature_ev > 0.0
        assert type(self.jet.number_density) is float
        assert self.jet.number_density > 0.0
        self.layout.validate()


class ArgonJets(JetApplication):
    config_type = ArgonConfig

    @property
    @override
    def bounds(self):
        return -2.0, 2.0, -2.0, 2.0, -2.0, 2.0

    @property
    def dx(self):
        return 2.0 * self.config.jet.radius / self.nx

    def _single_jet(self):
        jet = self.config.jet
        local_bounds = (
            0.0,
            jet.length,
            -jet.radius,
            jet.radius,
            -jet.radius,
            jet.radius,
        )
        axial, cross_y, cross_z = close_packed_lattice(local_bounds, self.dx)
        inside = cross_y * cross_y + cross_z * cross_z < jet.radius**2
        return axial[inside], cross_y[inside], cross_z[inside]

    @staticmethod
    def _velocity_field(axial, cross_y, cross_z, jet):
        radius = np.sqrt(cross_y * cross_y + cross_z * cross_z)
        inverse_radius = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
        radial_ratio = radius / jet.radius
        radial_remaining = 1.0 - radial_ratio * radial_ratio
        jet_length = jet.length
        taper_length = jet.end_taper_fraction * jet_length
        axial_local = 0.5 * jet_length - axial
        axial_plus = (axial_local + 0.5 * jet_length) / taper_length
        axial_minus = (axial_local - 0.5 * jet_length) / taper_length
        axial_envelope = 0.5 * (np.tanh(axial_plus) - np.tanh(axial_minus))
        toroidal_peak_shape = 216.0 / (343.0 * np.sqrt(7.0))
        azimuthal_velocity = jet.azimuthal_speed * radial_ratio * radial_remaining**3 / toroidal_peak_shape * axial_envelope
        cross_y_velocity = azimuthal_velocity * cross_z * inverse_radius
        cross_z_velocity = -azimuthal_velocity * cross_y * inverse_radius
        return np.vstack((cross_z_velocity, cross_y_velocity, np.full(len(axial), jet.axial_speed)))

    @override
    def create_mhd_particles(self):
        axial, cross_y, cross_z = self._single_jet()
        jet = self.config.jet
        positions = []
        velocities = []
        local_position = np.vstack((cross_z, cross_y, -axial))
        local_velocity = self._velocity_field(axial, cross_y, cross_z, jet)
        fronts, directions = latitude_rings(self.config.layout)
        for front, direction in zip(fronts, directions, strict=True):
            basis = jet_basis(direction)
            positions.append(basis @ local_position + front[:, None])
            velocities.append(basis @ local_velocity)

        x, y, z = np.concatenate(positions, axis=1)
        u, v, w = np.concatenate(velocities, axis=1)
        count_per_jet = len(axial)
        count = len(x)
        rho = jet.number_density * ARGON_MASS_NUMBER * ATOMIC_MASS
        temperature = jet.temperature_ev * EV_TO_K
        pressure = rho * temperature * GAS_CONSTANT / ARGON_MOLAR_MASS
        energy = pressure / ((self.gamma - 1.0) * rho)
        particle_volume = pi * jet.radius**2 * jet.length / count_per_jet
        mass = rho * particle_volume
        particle = get_particle_array(
            name="fluid",
            x=x,
            y=y,
            z=z,
            u=u,
            v=v,
            w=w,
            rho=np.full(count, rho),
            h=np.full(count, self.hfact * particle_volume ** (1.0 / 3.0)),
            m=np.full(count, mass),
            e=np.full(count, energy),
            gid=np.arange(count, dtype=np.uint32),
            Bx=np.zeros(count),
            By=np.zeros(count),
            Bz=np.zeros(count),
            alpha1=np.zeros(count),
        )
        particle.add_property("material_id", type="int", data=np.full(count, ARGON_MATERIAL_ID, dtype=np.int32))
        particle.add_property("molar_mass", data=np.full(count, ARGON_MOLAR_MASS))
        particle.add_property("source_id", type="int", data=np.repeat(np.arange(1, len(fronts) + 1, dtype=np.int32), count_per_jet))
        return [particle]
