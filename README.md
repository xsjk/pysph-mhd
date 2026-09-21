# PySPH MHD

A smoothed particle hydrodynamics (SPH) solver for magnetohydrodynamics (MHD), built on PySPH with CUDA acceleration.

## Installation

```bash
pip install 'pysph[cuda13x] @ git+https://github.com/xsjk/pysph.git'
```

Commands assume the default `python` has the required dependencies and CUDA support.

## Run

```bash
python -m src.run configs/alfven.toml
```

Standard MHD benchmarks are configured in `configs/`: `alfven`, `jadvect`,
`mhdblast`, `mhdrotor`, `mhdshock`, `mhdsine`, `mhdvortex`, `mhdwave`, and `orstang`.
