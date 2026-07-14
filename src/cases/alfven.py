from typing import override

import numpy as np
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array

from ..application import MHDApplication


class AlfvenWave(MHDApplication):
    gamma: float
    kernel: str
    density: str
    hfact: float
    tf: float
    pfreq: int
    nx: int
    periodic_mode: str

    @override
    def initialize(self):
        super().initialize()
        self.gamma = 5.0 / 3.0
        self.kernel = "quintic"
        self.hfact = 1.0
        self.tf = 1.0
        self.pfreq = 20
        self.nx = 64
        self.periodic_mode = "ghost"

    @override
    def add_user_options(self, group):
        super().add_user_options(group)
        group.add_argument("--nx", type=int, default=self.nx)
        group.add_argument("--periodic-mode", choices=("ghost", "minimum_image"), default=self.periodic_mode)

    @override
    def consume_user_options(self):
        super().consume_user_options()
        self.nx = self.options.nx
        self.periodic_mode = self.options.periodic_mode
        assert self.nx > 0

    @property
    def dx(self):
        return 3.0 / self.nx

    @override
    def create_domain(self):
        return DomainManager(
            xmin=-1.5,
            xmax=1.5,
            ymin=-0.75,
            ymax=0.75,
            zmin=-0.75,
            zmax=0.75,
            periodic_in_x=True,
            periodic_in_y=True,
            periodic_in_z=True,
            periodic_mode=self.periodic_mode,
        )

    @override
    def create_mhd_particles(self):
        ny = 2 * int(1.5 / (self.dx * np.sqrt(3.0 / 4.0)) / 2)
        nz = 3 * int((int(1.5 / (self.dx * np.sqrt(6.0) / 3.0)) + 1) / 3)
        dy = 1.5 / ny
        dz = 1.5 / nz
        k, layer_y, layer_z = np.meshgrid(np.arange(self.nx), np.arange(1, ny + 1), np.arange(1, nz + 1), indexing="ij")
        layer_y_parity = layer_y % 2
        layer_z_modulo = layer_z % 3
        x_offset = np.where((layer_z_modulo == 0) & (layer_y_parity == 0), 0.5, 0.0)
        x_offset = np.where((layer_z_modulo == 2) & (layer_y_parity == 1), 0.5, x_offset)
        x_offset = np.where((layer_z_modulo == 1) & (layer_y_parity == 0), 0.5, x_offset)
        y_offset = np.where(layer_z_modulo == 0, 2.0 / 3.0, np.where(layer_z_modulo == 2, 1.0 / 3.0, 0.0))
        x = (-1.5 + (0.25 + k + x_offset) * self.dx).ravel().astype(np.float32).astype(float)
        y = (-0.75 + (layer_y - 1.0 + 1.0 / 6.0 + y_offset) * dy).ravel().astype(np.float32).astype(float)
        z = (-0.75 + (layer_z - 0.5) * dz).ravel().astype(np.float32).astype(float)
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


if __name__ == "__main__":
    AlfvenWave().run()
