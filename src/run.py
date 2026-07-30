import argparse
from pathlib import Path

from .cases.alfven import AlfvenWave
from .cases.jadvect import CurrentLoopAdvection
from .cases.jet import JMXJets
from .cases.mhdblast import MHDBlast
from .cases.mhdrotor import MHDRotor
from .cases.mhdshock import MHDShock
from .cases.mhdsine import MHDSine
from .cases.mhdvortex import MHDVortex
from .cases.mhdwave import MHDWave
from .cases.orstang import OrszagTang
from .config import load_config, pysph_arguments

CASES = {
    "1jet": JMXJets,
    "3jet": JMXJets,
    "12jet": JMXJets,
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
    arguments = parser.parse_args()
    config_types = {name: application_type.config_type for name, application_type in CASES.items()}
    config = load_config(arguments.config, config_types)
    application = CASES[config.case.name](config)
    application.run(pysph_arguments(config.execution))


if __name__ == "__main__":
    main()
