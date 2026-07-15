from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class MHDShock(MHDApplication):
    @override
    def initialize(self):
        super().initialize()
        self.gamma = 2.0
        self.kernel = "quintic"
        self.hfact = 1.0
        self.tf = 0.1
        self.pfreq = 20
        self.nx = 256

    @override
    def consume_user_options(self):
        super().consume_user_options()
        assert self.nx % 2 == 0

    @property
    @override
    def bounds(self):
        dx_left = 0.5 / self.nx
        dx_right = 2.0 * dx_left
        ymax = 6.0 * dx_right * np.sqrt(3.0 / 4.0)
        zmax = 6.0 * dx_right * np.sqrt(6.0) / 3.0
        return -0.5 - 1000.0 * dx_left, 0.5 + 1000.0 * dx_right, -ymax, ymax, -zmax, zmax

    @override
    def create_mhd_particles(self):
        _, _, ymin, ymax, zmin, zmax = self.bounds
        dx_left = 0.5 / self.nx
        dx_right = 2.0 * dx_left
        left_bounds = (-0.5, 0.0, ymin, ymax, zmin, zmax)
        right_bounds = (0.0, 0.5, ymin, ymax, zmin, zmax)
        left = close_packed_lattice(left_bounds, dx_left)
        right = close_packed_lattice(right_bounds, dx_right)
        x = np.concatenate((left[0], right[0]))
        y = np.concatenate((left[1], right[1]))
        z = np.concatenate((left[2], right[2]))
        count = len(x)
        is_left = x <= 0.0
        rho = np.where(is_left, 1.0, 0.125)
        pressure = np.where(is_left, 1.0, 0.1)
        mass = 0.5 * (ymax - ymin) * (zmax - zmin) / len(left[0])
        particle_type = np.where(
            (x < -0.5 + 6.0 * dx_left) | (x > 0.5 - 6.0 * dx_right),
            3,
            1,
        ).astype(np.int32)
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=np.zeros(count),
                v=np.zeros(count),
                w=np.zeros(count),
                rho=rho,
                h=self.hfact * (mass / rho) ** (1.0 / 3.0),
                m=np.full(count, mass),
                e=pressure / ((self.gamma - 1.0) * rho),
                Bx=np.full(count, 0.75),
                By=np.where(is_left, 1.0, -1.0),
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
                itype=particle_type,
            ),
        ]


if __name__ == "__main__":
    MHDShock().run()
