from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class MHDSine(MHDApplication):
    @property
    @override
    def bounds(self):
        zmax = 4.0 * np.sqrt(6.0) / self.nx
        return -0.5, 0.5, -0.5, 0.5, -zmax, zmax

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        x, y, z = close_packed_lattice(self.bounds, (xmax - xmin) / self.nx)
        count = len(x)
        rho = 2.0
        mass = rho * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / count
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=np.zeros(count),
                v=np.zeros(count),
                w=np.zeros(count),
                rho=np.full(count, rho),
                h=np.full(count, self.hfact * (mass / rho) ** (1.0 / 3.0)),
                m=np.full(count, mass),
                e=np.full(count, 7.5),
                Bx=np.zeros(count),
                By=np.sin(2.0 * np.pi * (x - xmin)),
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
            ),
        ]
