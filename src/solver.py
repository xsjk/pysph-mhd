import logging
from typing import TYPE_CHECKING, override

from compyle.profile import profile_ctx
from pysph.solver.solver import EPSILON, Solver
from pysph.solver.utils import ProgressBar

if TYPE_CHECKING:
    from pysph.sph.integrator import Integrator

logger = logging.getLogger(__name__)


class MHDSolver(Solver):
    integrator: Integrator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_acceleration_is_current = False

    @override
    def solve(self, show_progress=True):
        show = False if self.in_parallel else show_progress
        bar = ProgressBar(self.t, self.tf, show=show)
        self._epsilon = EPSILON * self.tf

        self.dump_output()
        self.barrier()

        reorder_freq = self.reorder_freq
        if reorder_freq > 0:
            self.reorder_particles()

        # Skip the startup RHS when the application has already warmed it.
        if self.initial_acceleration_is_current:
            self.initial_acceleration_is_current = False
        else:
            self.integrator.initial_acceleration(self.t, self.dt)

        self.dt = self._get_timestep()

        while (self.tf - self.t) > self._epsilon and self.count < self.max_steps:
            if self.pre_step_callbacks:
                with profile_ctx("Solver.pre_step_callback"):
                    for callback in self.pre_step_callbacks:
                        callback(self)

            if self.rank == 0:
                logger.debug("Iteration=%d, time=%f, timestep=%f", self.count, self.t, self.dt)
            self.integrator.step(self.t, self.dt)

            if self.post_step_callbacks:
                with profile_ctx("Solver.post_step_callback"):
                    for callback in self.post_step_callbacks:
                        callback(self)

            self.t += self.dt
            self.count += 1
            self._epsilon = EPSILON * self.tf * self.count

            self.dt = self._get_timestep()

            self._dump_output_if_needed()

            bar.update(self.t)

            self.update_particle_time()

            if reorder_freq > 0 and (self.count % reorder_freq == 0):
                self.reorder_particles()

            if self.execute_commands is not None and self.count % self.command_interval == 0:
                self.execute_commands(self)

        bar.finish()

        self.dump_output()
