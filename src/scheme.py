from typing import override

from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.sph.equation import Group
from pysph.sph.integrator import PECIntegrator
from pysph.sph.scheme import Scheme

from .equations import (
    AdaptiveTimestep,
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
from .integrator import MHDStep
from .particles import BASE_PROPERTIES, OUTPUT_PROPERTIES, PROPERTIES
from .solver import MHDSolver

HFACT = 1.2
MAX_DENSITY_ITERATIONS = 250


class MHDScheme(Scheme):
    def __init__(self, gamma, kernel, density="iterate", hfact=HFACT):
        assert gamma > 1.0
        assert kernel in {"cubic", "quintic"}
        assert density in {"iterate", "single"}
        self.gamma = gamma
        self.kernel_name = kernel
        self.density = density
        self.hfact = hfact
        self.solver = None

    @override
    def configure(self, **kw):
        assert tuple(kw) == ("density",)
        density = kw["density"]
        assert density in {"iterate", "single"}
        self.density = density

    @override
    def configure_solver(self, **kw):
        kernel = CubicSpline(dim=3) if self.kernel_name == "cubic" else QuinticSpline(dim=3)
        integrator = PECIntegrator(fluid=MHDStep(self.gamma))
        self.solver = MHDSolver(dim=3, integrator=integrator, kernel=kernel, **kw)

    @override
    def setup_properties(self, particle):
        self._ensure_properties(particle, list(BASE_PROPERTIES + PROPERTIES), clean=True)
        particle.set_output_arrays(list(OUTPUT_PROPERTIES))

    @override
    def get_equations(self):
        sources = ["fluid"]
        iterate = self.density == "iterate"
        return [
            Group(equations=[DensityIteration(dest="fluid", sources=sources, k=self.hfact, dim=3, iterate_only_once=not iterate)], update_nnps=True, iterate=iterate, min_iterations=2 if iterate else 0, max_iterations=MAX_DENSITY_ITERATIONS if iterate else 1),
            Group(equations=[SmoothingLengthRateFromDensityRelation(dest="fluid", sources=None, k=self.hfact, dim=3), DensityFromSmoothingLength(dest="fluid", sources=None, k=self.hfact, dim=3)]),
            Group(equations=[IdealGasEOS(dest="fluid", sources=None, gamma=self.gamma)]),
            Group(equations=[CullenDehnenAlphaDiagnostics(dest="fluid", sources=sources)]),
            Group(equations=[StrainDiagnostics(dest="fluid", sources=None), MagneticStateReconstruction(dest="fluid", sources=None)]),
            Group(equations=[MHDAccelerations(dest="fluid", sources=sources)]),
            Group(equations=[SmoothingLengthRateFromForceDivergence(dest="fluid", sources=None, dim=3)]),
            Group(equations=[MagneticStressAccelerations(dest="fluid", sources=sources)]),
            Group(equations=[MagneticStateRates(dest="fluid", sources=sources)]),
            Group(equations=[EnergyLimiter(dest="fluid", sources=None), DivBCorrection(dest="fluid", sources=None), AdaptiveTimestep(dest="fluid", sources=None)]),
        ]
