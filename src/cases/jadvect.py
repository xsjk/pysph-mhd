from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class CurrentLoopAdvection(MHDApplication):
    @override
    def initialize(self):
        super().initialize()
        self.gamma = 5.0 / 3.0
        self.kernel = "cubic"
        self.hfact = 1.2
        self.tf = 1.0
        self.pfreq = 20
        self.nx = 128

    @property
    @override
    def bounds(self):
        zmax = 4.0 * np.sqrt(6.0) / self.nx
        return -1.0, 1.0, -0.5, 0.5, -zmax, zmax

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        x, y, z = close_packed_lattice(self.bounds, (xmax - xmin) / self.nx)
        count = len(x)
        rho = 1.0
        mass = rho * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / count
        loop_radius = 0.3
        radius = np.sqrt(x * x + y * y)
        inside = radius < loop_radius
        bx = np.zeros(count)
        by = np.zeros(count)
        bx[inside] = -0.001 * y[inside] / radius[inside]
        by[inside] = 0.001 * x[inside] / radius[inside]
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=np.full(count, 2.0),
                v=np.full(count, 1.0),
                w=np.full(count, 0.1 * np.sqrt(5.0)),
                rho=np.full(count, rho),
                h=np.full(count, self.hfact * (mass / rho) ** (1.0 / 3.0)),
                m=np.full(count, mass),
                e=np.full(count, 1.5),
                Bx=bx,
                By=by,
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
            ),
        ]


if __name__ == "__main__":
    CurrentLoopAdvection().run()
