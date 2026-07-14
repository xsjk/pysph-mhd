from typing import override

import numpy as np
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

    @override
    def initialize(self):
        self.density = "iterate"

    def create_mhd_particles(self):
        raise NotImplementedError

    @override
    def create_particles(self):
        particles = self.create_mhd_particles()
        assert len(particles) == 1
        assert particles[0].name == "fluid"
        self.scheme.setup_properties(particles[0])
        return particles

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

    @override
    def consume_user_options(self):
        self.density = self.options.density

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
