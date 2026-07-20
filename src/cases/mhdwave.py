from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class MHDWave(MHDApplication):
    @property
    @override
    def bounds(self):
        zmax = 4.0 * np.sqrt(6.0) / self.nx
        return -2.0, 2.0, -1.0, 1.0, -zmax, zmax

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        x, y, z = close_packed_lattice(self.bounds, (xmax - xmin) / self.nx)
        count = len(x)
        rho = 2.0
        mass = rho * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / count
        h = self.hfact * (mass / rho) ** (1.0 / 3.0)
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=0.01 * np.exp(-((x / (3.0 * h)) ** 2)),
                v=np.zeros(count),
                w=np.zeros(count),
                rho=np.full(count, rho),
                h=np.full(count, h),
                m=np.full(count, mass),
                e=np.full(count, 0.75),
                Bx=np.full(count, np.sqrt(2.0 / 3.0)),
                By=np.zeros(count),
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
            ),
        ]
