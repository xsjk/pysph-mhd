from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class MHDVortex(MHDApplication):
    @override
    def initialize(self):
        super().initialize()
        self.gamma = 5.0 / 3.0
        self.kernel = "cubic"
        self.hfact = 1.2
        self.tf = 10.0
        self.pfreq = 20
        self.nx = 64

    @property
    @override
    def bounds(self):
        zmax = 4.0 * np.sqrt(6.0) / self.nx
        return -5.0, 5.0, -5.0, 5.0, -zmax, zmax

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        x, y, z = close_packed_lattice(self.bounds, (xmax - xmin) / self.nx)
        count = len(x)
        rho = 1.0
        mass = rho * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / count
        factor = np.exp(0.5 * (1.0 - x * x - y * y)) / (2.0 * np.pi)
        pressure = 1.0 + np.exp(1.0 - x * x - y * y) * (0.5 - x * x - y * y) / (32.0 * np.pi**3)
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=1.0 - y * factor,
                v=1.0 + x * factor,
                w=np.ones(count),
                rho=np.full(count, rho),
                h=np.full(count, self.hfact * (mass / rho) ** (1.0 / 3.0)),
                m=np.full(count, mass),
                e=pressure / (rho * (self.gamma - 1.0)),
                Bx=-y * factor,
                By=x * factor,
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
            ),
        ]


if __name__ == "__main__":
    MHDVortex().run()
