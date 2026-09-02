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
DEUTERIUM_MASS_NUMBER = 2.0141
DEUTERIUM_MOLAR_MASS = 0.0020141
ATOMIC_MASS = 1.6726231e-27
EV_TO_K = 11604.505
GAS_CONSTANT = 8.31
SPHEROMAK_FUNDAMENTAL_ROOT = 4.493409457909064

ARGON_MATERIAL_ID = 0
DEUTERIUM_MATERIAL_ID = 1

JET_ANGLES = {
    "1jet": (0.0,),
    "3jet": (0.0, 30.0, -30.0),
    "12jet": tuple(30.0 * index for index in range(12)),
}


def jet_geometry_name(case_name):
    return case_name.removesuffix("_target")


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
        assert jet_geometry_name(self.case.name) in JET_ANGLES
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


@dataclass(frozen=True)
class TargetConfig:
    radius: float
    temperature_ev: float
    number_density: float
    central_field: float
    lambda_value: float
    magnetic_handedness: float

    def validate(self):
        assert type(self.radius) is float
        assert self.radius > 0.0
        assert type(self.temperature_ev) is float
        assert self.temperature_ev > 0.0
        assert type(self.number_density) is float
        assert self.number_density > 0.0
        assert type(self.central_field) is float
        assert self.central_field > 0.0
        assert type(self.lambda_value) is float
        assert self.lambda_value > 0.0
        assert np.isclose(self.lambda_value * self.radius, SPHEROMAK_FUNDAMENTAL_ROOT)
        assert type(self.magnetic_handedness) is float
        assert np.isclose(abs(self.magnetic_handedness), 1.0)


@dataclass(frozen=True)
class TargetJetSimulationConfig(JetSimulationConfig):
    target: TargetConfig

    @override
    def validate(self):
        super().validate()
        assert self.case.name.endswith("_target")
        self.target.validate()


def spherical_spheromak_field(x, y, z, target):
    assert x.shape == y.shape == z.shape
    radius = np.sqrt(x * x + y * y + z * z)
    cylindrical_radius = np.sqrt(x * x + z * z)
    inside = radius < target.radius
    dimensionless_radius = target.lambda_value * radius
    q2 = dimensionless_radius * dimensionless_radius
    regular = dimensionless_radius > 1.0e-4

    j0 = 1.0 - q2 / 6.0 + q2 * q2 / 120.0
    np.divide(
        np.sin(dimensionless_radius),
        dimensionless_radius,
        out=j0,
        where=regular,
    )
    j1_over_q = 1.0 / 3.0 - q2 / 30.0 + q2 * q2 / 840.0
    regular_j1 = np.zeros_like(radius)
    np.divide(
        np.sin(dimensionless_radius),
        dimensionless_radius**3,
        out=regular_j1,
        where=regular,
    )
    regular_cosine = np.zeros_like(radius)
    np.divide(
        np.cos(dimensionless_radius),
        dimensionless_radius**2,
        out=regular_cosine,
        where=regular,
    )
    j1_over_q[regular] = regular_j1[regular] - regular_cosine[regular]
    j1 = dimensionless_radius * j1_over_q
    j1_derivative = j0 - 2.0 * j1_over_q

    inverse_radius = np.zeros_like(radius)
    np.divide(1.0, radius, out=inverse_radius, where=radius > 0.0)
    inverse_cylindrical_radius = np.zeros_like(radius)
    np.divide(
        1.0,
        cylindrical_radius,
        out=inverse_cylindrical_radius,
        where=cylindrical_radius > 0.0,
    )
    cosine_theta = y * inverse_radius
    sine_theta = cylindrical_radius * inverse_radius
    cosine_phi = x * inverse_cylindrical_radius
    sine_phi = -z * inverse_cylindrical_radius

    coefficient = 1.5 * target.central_field
    radial_field = 2.0 * coefficient * j1_over_q * cosine_theta
    polar_field = -coefficient * (j1_over_q + j1_derivative) * sine_theta
    azimuthal_field = target.magnetic_handedness * coefficient * j1 * sine_theta

    bx = radial_field * x * inverse_radius + polar_field * cosine_theta * cosine_phi - azimuthal_field * sine_phi
    by = radial_field * cosine_theta - polar_field * sine_theta
    bz = radial_field * z * inverse_radius - polar_field * cosine_theta * sine_phi - azimuthal_field * cosine_phi
    origin = radius <= np.finfo(radius.dtype).eps
    bx[origin] = 0.0
    by[origin] = target.central_field
    bz[origin] = 0.0
    return (
        np.where(inside, bx, 0.0),
        np.where(inside, by, 0.0),
        np.where(inside, bz, 0.0),
    )


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
        for angle_degrees in JET_ANGLES[jet_geometry_name(self.config.case.name)]:
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
        pressure = rho * temperature * GAS_CONSTANT / ARGON_MOLAR_MASS
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


class JMXJetsTarget(JMXJets):
    config_type = TargetJetSimulationConfig

    def _create_target(self):
        target = self.config.target
        bounds = (
            -target.radius,
            target.radius,
            -target.radius,
            target.radius,
            -target.radius,
            target.radius,
        )
        x, y, z = close_packed_lattice(bounds, self.dx)
        inside = x * x + y * y + z * z < target.radius**2
        x, y, z = x[inside], y[inside], z[inside]
        count = len(x)
        rho = target.number_density * DEUTERIUM_MASS_NUMBER * ATOMIC_MASS
        temperature = target.temperature_ev * EV_TO_K
        pressure = rho * temperature * GAS_CONSTANT / DEUTERIUM_MOLAR_MASS
        energy = pressure / ((self.gamma - 1.0) * rho)
        particle_volume = 4.0 * pi * target.radius**3 / (3.0 * count)
        bx, by, bz = spherical_spheromak_field(x, y, z, target)
        return get_particle_array(
            name="fluid",
            x=x,
            y=y,
            z=z,
            u=np.zeros(count),
            v=np.zeros(count),
            w=np.zeros(count),
            rho=np.full(count, rho),
            h=np.full(count, self.hfact * particle_volume ** (1.0 / 3.0)),
            m=np.full(count, rho * particle_volume),
            e=np.full(count, energy),
            Bx=bx,
            By=by,
            Bz=bz,
            alpha1=np.zeros(count),
        )

    @override
    def create_mhd_particles(self):
        jet = super().create_mhd_particles()[0]
        target = self._create_target()
        properties = ("x", "y", "z", "u", "v", "w", "rho", "h", "m", "e", "Bx", "By", "Bz", "alpha1")
        merged = {name: np.concatenate((getattr(jet, name), getattr(target, name))) for name in properties}
        jet_count = len(jet.x)
        target_count = len(target.x)
        particle = get_particle_array(
            name="fluid",
            gid=np.arange(jet_count + target_count, dtype=np.uint32),
            **merged,
        )
        particle.add_property(
            "material_id",
            type="int",
            data=np.concatenate(
                (
                    np.full(jet_count, ARGON_MATERIAL_ID, dtype=np.int32),
                    np.full(target_count, DEUTERIUM_MATERIAL_ID, dtype=np.int32),
                )
            ),
        )
        particle.add_property(
            "molar_mass",
            data=np.concatenate(
                (
                    np.full(jet_count, ARGON_MOLAR_MASS),
                    np.full(target_count, DEUTERIUM_MOLAR_MASS),
                )
            ),
        )
        return [particle]

    @override
    def create_particles(self):
        particles = super().create_particles()
        particle = particles[0]
        particle.set_output_arrays(
            [
                *particle.output_property_arrays,
                "material_id",
                "molar_mass",
                "divBsymm",
                "divBdiff",
            ]
        )
        return particles
