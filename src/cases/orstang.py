from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class OrszagTang(MHDApplication):
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
        return -0.5, 0.5, -0.5, 0.5, -zmax, zmax

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        x, y, z = close_packed_lattice(self.bounds, (xmax - xmin) / self.nx)
        count = len(x)
        magnetic_field = 1.0 / np.sqrt(4.0 * np.pi)
        pressure = 0.5 * magnetic_field * magnetic_field * (10.0 / 3.0)
        rho = self.gamma * pressure
        mass = rho * (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / count
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=-np.sin(2.0 * np.pi * (y - ymin)),
                v=np.sin(2.0 * np.pi * (x - xmin)),
                w=np.zeros(count),
                rho=np.full(count, rho),
                h=np.full(count, self.hfact * (mass / rho) ** (1.0 / 3.0)),
                m=np.full(count, mass),
                e=np.full(count, pressure / ((self.gamma - 1.0) * rho)),
                Bx=-magnetic_field * np.sin(2.0 * np.pi * (y - ymin)),
                By=magnetic_field * np.sin(4.0 * np.pi * (x - xmin)),
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
            ),
        ]


if __name__ == "__main__":
    OrszagTang().run()
