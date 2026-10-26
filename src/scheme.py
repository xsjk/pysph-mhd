from typing import override

from pysph.base.kernels import CubicSpline, Gaussian, QuinticSpline, SuperGaussian, WendlandQuintic, WendlandQuinticC4, WendlandQuinticC6
from pysph.sph.equation import Group
from pysph.sph.integrator import PECIntegrator
from pysph.sph.scheme import Scheme

from .equations import (
    AdaptiveTimestep,
    ArgonHydrogenEOS,
    CullenDehnenAlphaDiagnostics,
    DensityFromSmoothingLength,
    DensityIteration,
    DivBCorrection,
    EnergyLimiter,
    IdealGasEOS,
    MagneticStateRates,
    MagneticStateReconstruction,
    MagneticStressAccelerations,
    MHDAccelerations,
    SmoothingLengthRateFromDensityRelation,
    SmoothingLengthRateFromForceDivergence,
    StrainDiagnostics,
)
from .integrator import MHDStep, PlasmaMHDStep
from .particles import BASE_PROPERTIES, OUTPUT_PROPERTIES, PROPERTIES
from .solver import MHDSolver

MAX_DENSITY_ITERATIONS = 250


class MHDScheme(Scheme):
    def __init__(self, config):
        self.plasma_eos = config.physics.plasma_eos
        self.gamma = config.physics.gamma
        self.mu0 = config.physics.mu0
        self.kernel_name = config.numerics.kernel
        self.density = config.numerics.density
        self.hfact = config.numerics.hfact
        self.artificial_viscosity = config.physics.artificial_viscosity
        self.artificial_thermal_conductivity = config.physics.artificial_thermal_conductivity
        self.artificial_magnetic_dissipation = config.physics.artificial_magnetic_dissipation
        self.cleaning_speed_factor = config.numerics.cleaning_speed_factor
        self.cleaning_damping_factor = config.numerics.cleaning_damping_factor
        self.timestep_factor = config.solver.timestep_factor
        self.solver = None

    @override
    def configure_solver(self, **kw):
        kernel_type = {
            "cubic": CubicSpline,
            "quintic": QuinticSpline,
            "wendland_c2": WendlandQuintic,
            "wendland_c4": WendlandQuinticC4,
            "wendland_c6": WendlandQuinticC6,
            "gaussian": Gaussian,
            "super_gaussian": SuperGaussian,
        }[self.kernel_name]
        kernel = kernel_type(dim=3)
        stepper = PlasmaMHDStep() if self.plasma_eos else MHDStep(self.gamma)
        integrator = PECIntegrator(fluid=stepper)
        self.solver = MHDSolver(dim=3, integrator=integrator, kernel=kernel, **kw)

    @override
    def setup_properties(self, particle):
        self._ensure_properties(particle, list(BASE_PROPERTIES + PROPERTIES), clean=True)
        particle.set_output_arrays(list(OUTPUT_PROPERTIES))
        if self.plasma_eos:
            assert ((particle.material_id == 0) | (particle.material_id == 2)).all(), "Plasma EOS requires Ar (0) or H (2)"
            from .plasma_eos import EOS_CONSTANTS  # ruff: ignore[import-outside-top-level] -- Load generated tables only for plasma cases.

            particle.add_property("T")
            particle.add_property("mean_charge")
            particle.add_property("eos_t_index", type="int", data=-1)
            for name, values in EOS_CONSTANTS.items():
                particle.add_constant(name, values)
            particle.set_output_arrays([*particle.output_property_arrays, "T", "mean_charge", "cs"])

    @override
    def get_equations(self):
        sources = ["fluid"]
        iterate = self.density == "iterate"
        return [
            Group(equations=[DensityIteration(dest="fluid", sources=sources, k=self.hfact, dim=3, iterate_only_once=not iterate)], update_nnps=True, iterate=iterate, min_iterations=2 if iterate else 0, max_iterations=MAX_DENSITY_ITERATIONS if iterate else 1),
            Group(equations=[SmoothingLengthRateFromDensityRelation(dest="fluid", sources=None, k=self.hfact, dim=3), DensityFromSmoothingLength(dest="fluid", sources=None, k=self.hfact, dim=3)]),
            Group(equations=[ArgonHydrogenEOS(dest="fluid", sources=None) if self.plasma_eos else IdealGasEOS(dest="fluid", sources=None, gamma=self.gamma)]),
            Group(equations=[CullenDehnenAlphaDiagnostics(dest="fluid", sources=sources)]),
            Group(equations=[StrainDiagnostics(dest="fluid", sources=None), MagneticStateReconstruction(dest="fluid", sources=None)]),
            Group(equations=[MHDAccelerations(dest="fluid", sources=sources, mu0=self.mu0, artificial_viscosity=self.artificial_viscosity, artificial_thermal_conductivity=self.artificial_thermal_conductivity)]),
            Group(equations=[SmoothingLengthRateFromForceDivergence(dest="fluid", sources=None, dim=3)]),
            Group(equations=[MagneticStressAccelerations(dest="fluid", sources=sources, mu0=self.mu0)]),
            Group(equations=[MagneticStateRates(dest="fluid", sources=sources, mu0=self.mu0, artificial_magnetic_dissipation=self.artificial_magnetic_dissipation, cleaning_speed_factor=self.cleaning_speed_factor, cleaning_damping_factor=self.cleaning_damping_factor)]),
            Group(equations=[EnergyLimiter(dest="fluid", sources=None, timestep_factor=self.timestep_factor), DivBCorrection(dest="fluid", sources=None, mu0=self.mu0), AdaptiveTimestep(dest="fluid", sources=None, timestep_factor=self.timestep_factor)]),
        ]
