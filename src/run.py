import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .cases.alfven import AlfvenWave
from .cases.argon import ArgonJets
from .cases.argon_hydrogen import ArgonHydrogenJets, ArgonPreflight
from .cases.hydrogen import HydrogenJets
from .cases.jadvect import CurrentLoopAdvection
from .cases.mhdblast import MHDBlast
from .cases.mhdrotor import MHDRotor
from .cases.mhdshock import MHDShock
from .cases.mhdsine import MHDSine
from .cases.mhdvortex import MHDVortex
from .cases.mhdwave import MHDWave
from .cases.orstang import OrszagTang
from .config import parse_config, pysph_arguments, read_config

JET_CASES = {
    "1_Ar": ArgonJets,
    "44_Ar": ArgonJets,
    "1_H": HydrogenJets,
    "2_H": HydrogenJets,
    "44_Ar_2_H": ArgonHydrogenJets,
}

CASES = {
    **JET_CASES,
    "alfven": AlfvenWave,
    "jadvect": CurrentLoopAdvection,
    "mhdblast": MHDBlast,
    "mhdrotor": MHDRotor,
    "mhdshock": MHDShock,
    "mhdsine": MHDSine,
    "mhdvortex": MHDVortex,
    "mhdwave": MHDWave,
    "orstang": OrszagTang,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("all", "preflight", "coupled"), default="all", help="Internal stages for 44_Ar_2_H.")
    arguments = parser.parse_args()
    values = read_config(arguments.config)
    application_type = CASES[values["case"]["name"]]
    config = parse_config(values, application_type.config_type)
    if config.case.name in JET_CASES and arguments.stage == "all":
        output = Path(config.execution.directory)
        assert not output.exists(), "Use a new output directory to preserve completed simulations."
        output.mkdir(parents=True)
        shutil.copyfile(arguments.config, output / "config.toml")
    if application_type is ArgonHydrogenJets:
        if arguments.stage == "all":
            for species in ("argon", "hydrogen"):
                shutil.copyfile(getattr(config, species).config, output / f"{species}_source.toml")
            for stage in ("preflight", "coupled"):
                with (output / f"{stage}.log").open("w") as log:
                    subprocess.run([sys.executable, "-m", "src.run", str(arguments.config), "--stage", stage], stdout=log, stderr=subprocess.STDOUT, check=True)
            return
        application_type = {"preflight": ArgonPreflight, "coupled": ArgonHydrogenJets}[arguments.stage]
    else:
        assert arguments.stage == "all", "Stages apply only to 44_Ar_2_H."
    application = application_type(config)
    application.run(pysph_arguments(application.config.execution))


if __name__ == "__main__":
    main()
