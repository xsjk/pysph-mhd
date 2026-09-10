import argparse
from pathlib import Path

from .cases.alfven import AlfvenWave
from .cases.jadvect import CurrentLoopAdvection
from .cases.jet import JMXJets, JMXJetsTarget
from .cases.mhdblast import MHDBlast
from .cases.mhdrotor import MHDRotor
from .cases.mhdshock import MHDShock
from .cases.mhdsine import MHDSine
from .cases.mhdvortex import MHDVortex
from .cases.mhdwave import MHDWave
from .cases.orstang import OrszagTang
from .config import parse_config, pysph_arguments, read_config

CASES = {
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
    values = read_config(arguments.config)
    if "jet" in values:
        application_type = JMXJetsTarget if "target" in values else JMXJets
        config = parse_config(values, application_type.config_type)
    else:
        application_type = CASES[values["case"]["name"]]
        config = parse_config(values, application_type.config_type)
    application = application_type(config)
    application.run(pysph_arguments(config.execution))


if __name__ == "__main__":
    main()
