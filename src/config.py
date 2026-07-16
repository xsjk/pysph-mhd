import tomllib
from dataclasses import dataclass
from pathlib import Path

CASE_NAMES = (
    "alfven",
    "jadvect",
    "mhdblast",
    "mhdrotor",
    "mhdshock",
    "mhdsine",
    "mhdvortex",
    "mhdwave",
    "orstang",
)


@dataclass(frozen=True)
class CaseConfig:
    name: str
    nx: int


@dataclass(frozen=True)
class PhysicsConfig:
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
    fused: bool
    directory: str


@dataclass(frozen=True)
class SimulationConfig:
    case: CaseConfig
    physics: PhysicsConfig
    numerics: NumericsConfig
    solver: SolverConfig
    execution: ExecutionConfig


def _assert_fields(values, fields):
    assert tuple(sorted(values)) == tuple(sorted(fields))


def load_config(path):
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)

    _assert_fields(values, ("case", "physics", "numerics", "solver", "execution"))
    case = values["case"]
    physics = values["physics"]
    numerics = values["numerics"]
    solver = values["solver"]
    execution = values["execution"]
    _assert_fields(case, ("name", "nx"))
    _assert_fields(physics, ("gamma", "mu0", "artificial_viscosity", "artificial_thermal_conductivity", "artificial_magnetic_dissipation"))
    _assert_fields(numerics, ("kernel", "hfact", "density", "periodic_mode", "cleaning_speed_factor", "cleaning_damping_factor"))
    _assert_fields(solver, ("tf", "pfreq", "adaptive_timestep", "timestep_factor"))
    _assert_fields(execution, ("backend", "fused", "directory"))

    config = SimulationConfig(
        case=CaseConfig(name=case["name"], nx=case["nx"]),
        physics=PhysicsConfig(gamma=physics["gamma"], mu0=physics["mu0"], artificial_viscosity=physics["artificial_viscosity"], artificial_thermal_conductivity=physics["artificial_thermal_conductivity"], artificial_magnetic_dissipation=physics["artificial_magnetic_dissipation"]),
        numerics=NumericsConfig(kernel=numerics["kernel"], hfact=numerics["hfact"], density=numerics["density"], periodic_mode=numerics["periodic_mode"], cleaning_speed_factor=numerics["cleaning_speed_factor"], cleaning_damping_factor=numerics["cleaning_damping_factor"]),
        solver=SolverConfig(tf=solver["tf"], pfreq=solver["pfreq"], adaptive_timestep=solver["adaptive_timestep"], timestep_factor=solver["timestep_factor"]),
        execution=ExecutionConfig(backend=execution["backend"], fused=execution["fused"], directory=execution["directory"]),
    )
    _validate_config(config)
    return config


def _validate_config(config):
    assert config.case.name in CASE_NAMES
    assert type(config.case.nx) is int
    assert config.case.nx > 0
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
    assert config.numerics.periodic_mode in {"ghost", "minimum_image"}
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
    return [*backend, *fused, "-d", execution.directory]
