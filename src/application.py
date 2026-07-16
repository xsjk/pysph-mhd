from typing import override

import numpy as np
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application

from .equations import COURANT_FACTOR, FORCE_FACTOR
from .particles import INITIAL_PREDICTOR_PAIRS, INITIAL_STATE_GPU_PROPERTIES
from .scheme import MHDScheme


class MHDApplication(Application):
    scheme: MHDScheme
    gamma: float
    kernel: str
    density: str
    hfact: float
    tf: float
    pfreq: int
    nx: int
    periodic_mode: str

    @override
    def initialize(self):
        self.density = "iterate"
        self.periodic_mode = "ghost"

    def create_mhd_particles(self):
        raise NotImplementedError

    def refresh_initial_thermodynamics(self, particle):
        pass

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        raise NotImplementedError

    @override
    def create_particles(self):
        particles = self.create_mhd_particles()
        assert len(particles) == 1
        assert particles[0].name == "fluid"
        self.scheme.setup_properties(particles[0])
        return particles

    @override
    def create_domain(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return DomainManager(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            periodic_in_x=True,
            periodic_in_y=True,
            periodic_in_z=True,
            periodic_mode=self.periodic_mode,
        )

    @override
    def create_scheme(self):
        assert all(hasattr(self, name) for name in ("gamma", "kernel", "hfact", "tf", "pfreq"))
        return MHDScheme(
            gamma=self.gamma,
            kernel=self.kernel,
            density=self.density,
            hfact=self.hfact,
        )

    @override
    def add_user_options(self, group):
        group.add_argument(
            "--density",
            choices=("iterate", "single"),
            default=self.density,
        )
        group.add_argument("--nx", type=int, default=self.nx)
        group.add_argument(
            "--periodic-mode",
            choices=("ghost", "minimum_image"),
            default=self.periodic_mode,
        )

    @override
    def consume_user_options(self):
        self.density = self.options.density
        self.nx = self.options.nx
        self.periodic_mode = self.options.periodic_mode
        assert self.nx > 0

    @override
    def configure_scheme(self):
        self.scheme.configure(density=self.density)
        self.scheme.configure_solver(
            dt=1.0,
            tf=self.tf,
            adaptive_timestep=True,
            pfreq=self.pfreq,
        )

    @override
    def solve(self):
        self._warm_initial_derivatives(self.solver)
        self._compute_initial_dt(self.solver)
        super().solve()

    @staticmethod
    def _sync_initial_mhd_state(particle):
        particle.Bevolx[:] = particle.Bx / particle.rho
        particle.Bevoly[:] = particle.By / particle.rho
        particle.Bevolz[:] = particle.Bz / particle.rho
        particle.psi[:] = 0.0
        for predicted, current in INITIAL_PREDICTOR_PAIRS:
            getattr(particle, predicted)[:] = getattr(particle, current)
        if particle.gpu is not None:
            particle.gpu.push(*INITIAL_STATE_GPU_PROPERTIES)

    def _warm_initial_derivatives(self, solver):
        particle = self.particles[0]
        solver.integrator.initial_acceleration(solver.t, solver.dt)
        if particle.gpu is not None:
            particle.gpu.pull("rho", "h")
        self.refresh_initial_thermodynamics(particle)
        if particle.gpu is not None:
            particle.gpu.push("e")
        self._sync_initial_mhd_state(particle)
        for _ in range(2):
            solver.integrator.initial_acceleration(solver.t, solver.dt)
        if particle.gpu is not None:
            particle.gpu.pull("rho", "h", "Bx", "By", "Bz")
        solver.initial_acceleration_is_current = True

    def _compute_initial_dt(self, solver):
        particle = self.particles[0]
        if particle.gpu is not None:
            particle.gpu.pull("h", "dt_cfl", "au", "av", "aw")
        force_norm = np.sqrt(particle.au * particle.au + particle.av * particle.av + particle.aw * particle.aw)
        assert np.all(particle.dt_cfl > 0.0)
        dt_courant = COURANT_FACTOR * particle.h / particle.dt_cfl
        dt_candidates = [float(np.min(dt_courant))]
        force_mask = force_norm > 0.0
        if np.any(force_mask):
            dt_force = FORCE_FACTOR * np.sqrt(particle.h[force_mask] / force_norm[force_mask])
            dt_candidates.append(float(np.min(dt_force)))
        solver.dt = min(dt_candidates)
        assert solver.dt > 0.0
