import tomllib
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import get_type_hints


@dataclass(frozen=True)
class CaseConfig:
    name: str
    nx: int


@dataclass(frozen=True)
class PhysicsConfig:
    plasma_eos: bool
    gamma: float
    mu0: float
    artificial_viscosity: float
    artificial_thermal_conductivity: float
    artificial_magnetic_dissipation: float


@dataclass(frozen=True)
class NumericsConfig:
    kernel: str
    hfact: float
    density: str
    periodic_mode: str
    cleaning_speed_factor: float
    cleaning_damping_factor: float


@dataclass(frozen=True)
class SolverConfig:
    tf: float
    pfreq: int
    adaptive_timestep: bool
    timestep_factor: float


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str
    nnps: str
    octree_leaf_size: int
    fused: bool
    directory: str


@dataclass(frozen=True)
class SimulationConfig:
    case: CaseConfig
    physics: PhysicsConfig
    numerics: NumericsConfig
    solver: SolverConfig
    execution: ExecutionConfig

    def validate(self):
        _validate_config(self)


def load_config(path, config_types):
    values = read_config(path)
    return parse_config(values, config_types[values["case"]["name"]])


def read_config(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def parse_config(values, config_type):
    config = _load_dataclass(config_type, values)
    config.validate()
    return config


def _load_dataclass(config_type, values):
    field_types = get_type_hints(config_type)
    assert values.keys() == field_types.keys()
    return config_type(**{name: _load_dataclass(field_type, values[name]) if is_dataclass(field_type) else values[name] for name, field_type in field_types.items()})


def _validate_config(config):
    assert type(config.case.name) is str
    assert config.case.name
    assert type(config.case.nx) is int
    assert config.case.nx > 0
    assert type(config.physics.plasma_eos) is bool
    assert type(config.physics.gamma) is float
    assert config.physics.gamma > 1.0
    assert type(config.physics.mu0) is float
    assert config.physics.mu0 > 0.0
    assert type(config.physics.artificial_viscosity) is float
    assert config.physics.artificial_viscosity >= 0.0
    assert type(config.physics.artificial_thermal_conductivity) is float
    assert config.physics.artificial_thermal_conductivity >= 0.0
    assert type(config.physics.artificial_magnetic_dissipation) is float
    assert config.physics.artificial_magnetic_dissipation >= 0.0
    assert config.numerics.kernel in {"cubic", "quintic", "wendland_c2", "wendland_c4", "wendland_c6", "gaussian", "super_gaussian"}
    assert type(config.numerics.hfact) is float
    assert config.numerics.hfact > 0.0
    assert config.numerics.density in {"iterate", "single"}
    assert config.numerics.periodic_mode in {"none", "ghost", "minimum_image"}
    assert type(config.numerics.cleaning_speed_factor) is float
    assert config.numerics.cleaning_speed_factor >= 0.0
    assert type(config.numerics.cleaning_damping_factor) is float
    assert config.numerics.cleaning_damping_factor >= 0.0
    assert type(config.solver.tf) is float
    assert config.solver.tf > 0.0
    assert type(config.solver.pfreq) is int
    assert config.solver.pfreq > 0
    assert type(config.solver.adaptive_timestep) is bool
    assert type(config.solver.timestep_factor) is float
    assert config.solver.timestep_factor > 0.0
    assert config.execution.backend in {"cython", "cuda", "opencl"}
    assert type(config.execution.nnps) is str
    assert type(config.execution.octree_leaf_size) is int
    assert config.execution.octree_leaf_size > 0
    assert type(config.execution.fused) is bool
    assert not config.execution.fused or config.execution.backend == "cuda"
    assert type(config.execution.directory) is str
    assert config.execution.directory


def pysph_arguments(execution):
    backend = {
        "cython": (),
        "cuda": ("--cuda",),
        "opencl": ("--opencl",),
    }[execution.backend]
    fused = {
        False: (),
        True: ("--fused",),
    }[execution.fused]
    return [*backend, *fused, "--nnps", execution.nnps, "--octree-leaf-size", str(execution.octree_leaf_size), "-d", execution.directory]
