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

## Ar/H jets with ionization

Generate the local EOS table before the first run with ionization:

```bash
python -m src.generate_plasma_eos \
  --argon-levels ../spmhd/spmhd/oneFluid/sph/plasmaProperties/plasmaProperties.py \
  --output src/data
python -m src.run configs/jet/44_Ar_2_H.toml
```

Every simulation TOML explicitly selects its EOS in `[physics]`:

```toml
[physics]
plasma_eos = true

[execution]
backend = "cuda"
fused = false
```

`true` enables the single-temperature LTE Saha-Boltzmann model;
`false` selects the ideal-gas EOS without loading a table. The option controls
both initial energy and time integration, independently of the case name.
The combined Ar/H configuration passes this option to both species, including
the Ar preflight stage. The supplied jet configs enable it; the MHD benchmarks
disable it. Ideal jets use neutral Ar and fixed, fully ionized H.
The supplied jet configs use CUDA with `fused = false`; the fused backend
does not yet support the static lookup-table arguments.

With ionization enabled, internal energy includes translation, ionization and bound-level
excitation. Each material solves its own charge neutrality; Ar and H do not
share an electron bath.

The generated `src/data/plasma_eos.npz` covers nucleus number densities
from 1e14 to 1e28 m^-3 and temperatures from 0.001 to 1e5 eV. Lookup is bilinear
in linear number density and temperature. Energy is inverted for temperature;
pressure and adiabatic sound speed follow the same interpolant. States outside
the table trigger a CUDA device assertion, without a per-step host error check. Saved `T` is
in kelvin and `mean_charge` is the average ion charge.
The video reader requires the saved `T` and `material_id` fields.

The generator needs the legacy `spmhd` source file above, but no existing table
or atomic-data JSON. It extracts the NIST-attributed Ar I-VI bound levels from
that file; higher Ar charge states use ground weights. The script contains
the Ar ionization thresholds from [Saloman (2010)](https://srd.nist.gov/jpcrdreprint/1.3337661.pdf)
and the 25 H levels through n=5 from the [NIST Handbook](https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm).
Generated data stay in the ignored `src/data/` directory. The model omits
radiation, non-LTE kinetics and density-dependent nonideal effects.
