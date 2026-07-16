from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class AlfvenWave(MHDApplication):
    @property
    @override
    def bounds(self):
        return -1.5, 1.5, -0.75, 0.75, -0.75, 0.75

    @property
    def dx(self):
        return 3.0 / self.nx

    @override
    def create_mhd_particles(self):
        x, y, z = close_packed_lattice(self.bounds, self.dx)
        x = x.astype(np.float32).astype(float)
        y = y.astype(np.float32).astype(float)
        z = z.astype(np.float32).astype(float)
        phase = 2.0 * np.pi * (x + 1.5)
        by = 0.1 * np.sin(phase)
        bz = 0.1 * np.cos(phase)
        count = len(x)
        mass = float(np.float32(6.75 / count))
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
                h=np.full(count, self.hfact * mass ** (1.0 / 3.0)),
                m=np.full(count, mass),
                e=np.full(count, 0.15),
                Bx=np.ones(count),
                By=by,
                Bz=bz,
                alpha1=np.zeros(count),
            ),
        ]
