"""Launch Ar first, hold H at its configured guns, then evolve both species."""

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import override

import h5py
import numpy as np
from pysph.base.utils import get_particle_array

from ..config import SimulationConfig, parse_config, read_config
from .argon import ArgonConfig, ArgonJets
from .hydrogen import HydrogenConfig, HydrogenJets
from .jets import JetApplication, latitude_rings

STATE_FIELDS = ("x", "y", "z", "u", "v", "w", "rho", "h", "m", "e", "Bx", "By", "Bz", "alpha1")


@dataclass(frozen=True)
class LaunchConfig:
    config: str
    nx: int
    axial_speed: float
    launch_time: float

    def validate(self):
        assert type(self.config) is str
        assert type(self.nx) is int
        assert self.nx > 0
        assert isfinite(self.axial_speed)
        assert self.axial_speed > 0.0
        assert isfinite(self.launch_time)
        assert self.launch_time >= 0.0


@dataclass(frozen=True)
class ArgonHydrogenConfig(SimulationConfig):
    argon: LaunchConfig
    hydrogen: LaunchConfig

    @override
    def validate(self):
        super().validate()
        self.argon.validate()
        self.hydrogen.validate()
        np.testing.assert_equal(self.argon.launch_time, 0.0)
        assert self.argon.launch_time < self.hydrogen.launch_time < self.solver.tf


def species_config(config, species):
    launch = getattr(config, species)
    config_type = {"argon": ArgonConfig, "hydrogen": HydrogenConfig}[species]
    original = parse_config(read_config(launch.config), config_type)
    physics = replace(original.physics, plasma_eos=config.physics.plasma_eos)
    assert physics == config.physics
    assert original.numerics == config.numerics
    return replace(
        original,
        case=replace(original.case, nx=launch.nx),
        physics=physics,
        jet=replace(original.jet, axial_speed=launch.axial_speed),
        solver=config.solver,
        execution=config.execution,
    )


def argon_preflight_config(config):
    argon = species_config(config, "argon")
    hydrogen = species_config(config, "hydrogen")
    np.testing.assert_allclose(argon.layout.radius, hydrogen.layout.radius, rtol=0.0, atol=1.0e-12)
    return replace(
        argon,
        solver=replace(config.solver, tf=config.hydrogen.launch_time, pfreq=1),
        execution=replace(config.execution, directory=str(Path(config.execution.directory) / "argon_preflight")),
    )


class ArgonPreflight(ArgonJets):
    """Evolve Ar while a separate, noninteracting H array is held at its guns."""

    def __init__(self, config):
        self.launch_config = config
        super().__init__(argon_preflight_config(config))

    @override
    def create_particles(self):
        particles = super().create_particles()
        hydrogen = HydrogenJets(species_config(self.launch_config, "hydrogen")).create_particles()[0]
        hydrogen.set_name("hydrogen")
        hydrogen.u[:], hydrogen.v[:], hydrogen.w[:] = 0.0, 0.0, 0.0
        if hydrogen.gpu is not None:
            hydrogen.gpu.push("u", "v", "w")
        # The scheme integrates only "fluid"; this array is stored but held fixed.
        return [*particles, hydrogen]


class ArgonHydrogenJets(JetApplication):
    config_type = ArgonHydrogenConfig

    @property
    @override
    def bounds(self):
        configs = [species_config(self.config, species) for species in ("argon", "hydrogen")]
        extent = max(config.layout.radius + config.jet.length for config in configs)
        return (-extent, extent) * 3

    @override
    def create_mhd_particles(self):
        argon_config = argon_preflight_config(self.config)
        paths = sorted(Path(argon_config.execution.directory).glob(f"{argon_config.case.name}_*.hdf5"))
        assert paths, "Run the Ar preflight stage before releasing hydrogen."
        species = []
        with h5py.File(paths[-1]) as handle:
            np.testing.assert_allclose(handle["solver_data"].attrs["t"], self.config.hydrogen.launch_time, rtol=1.0e-6, atol=0.0)
            for name in ("fluid", "hydrogen"):
                arrays = handle[f"particles/{name}/arrays"]
                order = np.argsort(arrays["gid"][...])
                np.testing.assert_array_equal(arrays["gid"][...][order], np.arange(len(order)))
                species.append({name: arrays[name][...][order] for name in (*STATE_FIELDS, "source_id", "material_id")})
        argon, hydrogen = species
        for name in ("u", "v", "w"):
            np.testing.assert_array_equal(hydrogen[name], 0.0)
        _, directions = latitude_rings(species_config(self.config, "hydrogen").layout)
        assert np.all((hydrogen["source_id"] >= 1) & (hydrogen["source_id"] <= len(directions))), "source_id must identify a configured H gun."
        velocity = self.config.hydrogen.axial_speed * directions[hydrogen["source_id"] - 1]
        hydrogen["u"], hydrogen["v"], hydrogen["w"] = velocity.T
        fields = {name: np.concatenate((argon[name], hydrogen[name])) for name in STATE_FIELDS}
        count = len(fields["x"])
        particle = get_particle_array(name="fluid", gid=np.arange(count, dtype=np.uint32), **fields)
        for name in ("source_id", "material_id"):
            particle.add_property(name, type="int", data=np.concatenate((argon[name], hydrogen[name])))
        return [particle]

    @override
    def configure_scheme(self):
        super().configure_scheme()
        # Continue the physical clock: all files use time since Ar launch.
        self.scheme.solver.t = self.config.hydrogen.launch_time
