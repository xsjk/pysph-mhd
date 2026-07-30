from dataclasses import dataclass
from math import cos, pi, sin
from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from ..config import SimulationConfig
from .lattice import close_packed_lattice

ARGON_MASS_NUMBER = 39.948
ARGON_MOLAR_MASS = 0.039948
ATOMIC_MASS = 1.6726231e-27
EV_TO_K = 11604.505

JET_ANGLES = {
    "1jet": (0.0,),
    "3jet": (0.0, 30.0, -30.0),
    "12jet": tuple(30.0 * index for index in range(12)),
}


@dataclass(frozen=True)
class JetConfig:
    inner_radius: float
    outer_radius: float
    radius: float
    end_taper_fraction: float
    axial_speed: float
    azimuthal_speed: float
    temperature_ev: float
    number_density: float
    axial_field: float
    toroidal_field: float


@dataclass(frozen=True)
class JetSimulationConfig(SimulationConfig):
    jet: JetConfig

    @override
    def validate(self):
        super().validate()
        assert self.case.name in JET_ANGLES
        assert type(self.jet.inner_radius) is float
        assert self.jet.inner_radius >= 0.0
        assert type(self.jet.outer_radius) is float
        assert self.jet.outer_radius > self.jet.inner_radius
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
        assert type(self.jet.axial_field) is float
        assert self.jet.axial_field >= 0.0
        assert type(self.jet.toroidal_field) is float
        assert self.jet.toroidal_field >= 0.0


class JMXJets(MHDApplication):
    config_type = JetSimulationConfig

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
            jet.inner_radius,
            jet.outer_radius,
            -jet.radius,
            jet.radius,
            -jet.radius,
            jet.radius,
        )
        axial, cross_y, cross_z = close_packed_lattice(local_bounds, self.dx)
        inside = cross_y * cross_y + cross_z * cross_z < jet.radius**2
        return axial[inside], cross_y[inside], cross_z[inside]

    @staticmethod
    def _magnetic_field(axial, cross_y, cross_z, cos_angle, sin_angle, jet):
        radius = np.sqrt(cross_y * cross_y + cross_z * cross_z)
        inverse_radius = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
        radial_ratio = radius / jet.radius
        radial_remaining = 1.0 - radial_ratio * radial_ratio
        jet_length = jet.outer_radius - jet.inner_radius
        taper_length = jet.end_taper_fraction * jet_length
        axial_local = 0.5 * (jet.inner_radius + jet.outer_radius) - axial
        axial_plus = (axial_local + 0.5 * jet_length) / taper_length
        axial_minus = (axial_local - 0.5 * jet_length) / taper_length
        axial_envelope = 0.5 * (np.tanh(axial_plus) - np.tanh(axial_minus))
        axial_derivative = 0.5 * (1.0 / np.cosh(axial_plus) ** 2 - 1.0 / np.cosh(axial_minus) ** 2) / taper_length
        axial_profile = radial_remaining**3 * (1.0 - 5.0 * radial_ratio**2)
        axial_field = jet.axial_field * axial_profile * axial_envelope
        radial_field = -0.5 * jet.axial_field * radius * radial_remaining**4 * axial_derivative
        toroidal_peak_shape = 216.0 / (343.0 * np.sqrt(7.0))
        toroidal_field = jet.toroidal_field * radial_ratio * radial_remaining**3 / toroidal_peak_shape * axial_envelope

        # Rotate the same divergence-free local helical field with each jet axis.
        cross_y_field = (radial_field * cross_y + toroidal_field * cross_z) * inverse_radius
        cross_z_field = (radial_field * cross_z - toroidal_field * cross_y) * inverse_radius
        return (
            -axial_field * cos_angle - cross_z_field * sin_angle,
            cross_y_field,
            -axial_field * sin_angle + cross_z_field * cos_angle,
        )

    @staticmethod
    def _velocity_field(axial, cross_y, cross_z, cos_angle, sin_angle, jet):
        radius = np.sqrt(cross_y * cross_y + cross_z * cross_z)
        inverse_radius = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
        radial_ratio = radius / jet.radius
        radial_remaining = 1.0 - radial_ratio * radial_ratio
        jet_length = jet.outer_radius - jet.inner_radius
        taper_length = jet.end_taper_fraction * jet_length
        axial_local = 0.5 * (jet.inner_radius + jet.outer_radius) - axial
        axial_plus = (axial_local + 0.5 * jet_length) / taper_length
        axial_minus = (axial_local - 0.5 * jet_length) / taper_length
        axial_envelope = 0.5 * (np.tanh(axial_plus) - np.tanh(axial_minus))
        toroidal_peak_shape = 216.0 / (343.0 * np.sqrt(7.0))
        azimuthal_velocity = jet.azimuthal_speed * radial_ratio * radial_remaining**3 / toroidal_peak_shape * axial_envelope
        cross_y_velocity = azimuthal_velocity * cross_z * inverse_radius
        cross_z_velocity = -azimuthal_velocity * cross_y * inverse_radius
        return (
            -jet.axial_speed * cos_angle - cross_z_velocity * sin_angle,
            cross_y_velocity,
            -jet.axial_speed * sin_angle + cross_z_velocity * cos_angle,
        )

    @override
    def create_mhd_particles(self):
        axial, cross_y, cross_z = self._single_jet()
        jet = self.config.jet
        positions = []
        velocities = []
        magnetic_fields = []
        for angle_degrees in JET_ANGLES[self.config.case.name]:
            angle = angle_degrees * pi / 180.0
            cos_angle = cos(angle)
            sin_angle = sin(angle)
            positions.append((
                axial * cos_angle - cross_z * sin_angle,
                cross_y,
                axial * sin_angle + cross_z * cos_angle,
            ))
            velocities.append(self._velocity_field(axial, cross_y, cross_z, cos_angle, sin_angle, jet))
            magnetic_fields.append(self._magnetic_field(axial, cross_y, cross_z, cos_angle, sin_angle, jet))

        x = np.concatenate([position[0] for position in positions])
        y = np.concatenate([position[1] for position in positions])
        z = np.concatenate([position[2] for position in positions])
        count_per_jet = len(axial)
        count = len(x)
        rho = jet.number_density * ARGON_MASS_NUMBER * ATOMIC_MASS
        temperature = jet.temperature_ev * EV_TO_K
        pressure = rho * temperature * 8.31 / ARGON_MOLAR_MASS
        energy = pressure / ((self.gamma - 1.0) * rho)
        particle_volume = pi * jet.radius**2 * (jet.outer_radius - jet.inner_radius) / count_per_jet
        mass = rho * particle_volume
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=np.concatenate([velocity[0] for velocity in velocities]),
                v=np.concatenate([velocity[1] for velocity in velocities]),
                w=np.concatenate([velocity[2] for velocity in velocities]),
                rho=np.full(count, rho),
                h=np.full(count, self.hfact * particle_volume ** (1.0 / 3.0)),
                m=np.full(count, mass),
                e=np.full(count, energy),
                gid=np.arange(count, dtype=np.uint32),
                Bx=np.concatenate([field[0] for field in magnetic_fields]),
                By=np.concatenate([field[1] for field in magnetic_fields]),
                Bz=np.concatenate([field[2] for field in magnetic_fields]),
                alpha1=np.zeros(count),
            ),
        ]
