from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class MHDBlast(MHDApplication):
    @property
    @override
    def bounds(self):
        return -0.5, 0.5, -0.5, 0.5, -0.5, 0.5

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        x, y, z = close_packed_lattice(self.bounds, (xmax - xmin) / self.nx)
        count = len(x)
        rho = 1.0
        mass = rho * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / count
        pressure = np.where(x * x + y * y + z * z < 0.125**2, 100.0, 1.0)
        magnetic_field = 10.0 / np.sqrt(2.0)
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
                e=pressure / (rho * (self.gamma - 1.0)),
                Bx=np.full(count, magnetic_field),
                By=np.zeros(count),
                Bz=np.full(count, magnetic_field),
                alpha1=np.zeros(count),
            ),
        ]
