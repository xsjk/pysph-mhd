from typing import override

import numpy as np
from pysph.base.utils import get_particle_array

from ..application import MHDApplication
from .lattice import close_packed_lattice


class MHDRotor(MHDApplication):
    @property
    @override
    def bounds(self):
        zmax = 2.0 * np.sqrt(6.0) / self.nx
        return -0.5, 0.5, -0.5, 0.5, -zmax, zmax

    @override
    def create_mhd_particles(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        spacing = (xmax - xmin) / self.nx
        disk_radius = 0.1
        outer = close_packed_lattice(self.bounds, spacing)
        outer_mask = outer[0] * outer[0] + outer[1] * outer[1] >= disk_radius**2
        inner = close_packed_lattice(self.bounds, spacing * 10.0 ** (-1.0 / 3.0))
        inner_mask = inner[0] * inner[0] + inner[1] * inner[1] <= disk_radius**2
        x = np.concatenate((outer[0][outer_mask], inner[0][inner_mask]))
        y = np.concatenate((outer[1][outer_mask], inner[1][inner_mask]))
        z = np.concatenate((outer[2][outer_mask], inner[2][inner_mask]))
        count = len(x)
        radius = np.sqrt(x * x + y * y)
        inside = radius <= disk_radius
        rho = np.where(inside, 10.0, 1.0)
        mass = ((xmax - xmin) * (ymax - ymin) + 10.0 * np.pi * disk_radius**2) * (zmax - zmin) / count
        velocity_factor = np.where(inside, 2.0 / disk_radius, 0.0)
        return [
            get_particle_array(
                name="fluid",
                x=x,
                y=y,
                z=z,
                u=-velocity_factor * y,
                v=velocity_factor * x,
                w=np.zeros(count),
                rho=rho,
                h=self.hfact * (mass / rho) ** (1.0 / 3.0),
                m=np.full(count, mass),
                e=1.0 / ((self.gamma - 1.0) * rho),
                Bx=np.full(count, 5.0 / np.sqrt(4.0 * np.pi)),
                By=np.zeros(count),
                Bz=np.zeros(count),
                alpha1=np.zeros(count),
            ),
        ]
