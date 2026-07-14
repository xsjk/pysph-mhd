from typing import override

import numpy as np
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array

from ..application import MHDApplication


class AlfvenWave(MHDApplication):
    gamma: float
    kernel: str
    density: str
    tf: float
    pfreq: int
    nx: int
    transverse_particles: int

    @override
    def initialize(self):
        super().initialize()
        self.gamma = 5.0 / 3.0
        self.kernel = "quintic"
        self.tf = 1.0
        self.pfreq = 20
        self.nx = 32
        self.transverse_particles = 6

    @override
    def add_user_options(self, group):
        super().add_user_options(group)
        group.add_argument("--nx", type=int, default=self.nx)
        group.add_argument(
            "--transverse-particles",
            type=int,
            default=self.transverse_particles,
        )

    @override
    def consume_user_options(self):
        super().consume_user_options()
        self.nx = self.options.nx
        self.transverse_particles = self.options.transverse_particles
        assert self.nx > 0
        assert self.transverse_particles > 0

    @property
    def dx(self):
        return 1.0 / self.nx

    @property
    def transverse_length(self):
        return self.transverse_particles * self.dx

    @override
    def create_domain(self):
        half_width = 0.5 * self.transverse_length
        return DomainManager(
            xmin=-0.5,
            xmax=0.5,
            ymin=-half_width,
            ymax=half_width,
            zmin=-half_width,
            zmax=half_width,
            periodic_in_x=True,
            periodic_in_y=True,
            periodic_in_z=True,
        )

    @override
    def create_mhd_particles(self):
        half_width = 0.5 * self.transverse_length
        x_axis = -0.5 + (np.arange(self.nx) + 0.5) * self.dx
        transverse_axis = (
            -half_width
            + (np.arange(self.transverse_particles) + 0.5) * self.dx
        )
        grid = np.meshgrid(
            x_axis,
            transverse_axis,
            transverse_axis,
            indexing="ij",
        )
        x, y, z = (values.ravel() for values in grid)
        phase = 2.0 * np.pi * (x + 0.5)
        by = 0.1 * np.sin(phase)
        bz = 0.1 * np.cos(phase)
        count = len(x)
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=np.zeros(count),
                v=by,
                w=bz,
                rho=np.ones(count),
                h=np.full(count, 1.2 * self.dx),
                m=np.full(count, self.dx**3),
                e=np.full(count, 0.15),
                Bx=np.ones(count),
                By=by,
                Bz=bz,
                alpha1=np.full(count, 0.1),
            )
        ]


if __name__ == "__main__":
    AlfvenWave().run()
