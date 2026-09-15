"""Replace [jet_generator] in a TOML config with explicit [[jets]]."""

import argparse
import tomllib
from math import cos, pi, sin, sqrt
from pathlib import Path


def planar(config):
    assert config["count"] > 0
    angles = (config["azimuth_start_degrees"] + index * config["azimuth_step_degrees"] for index in range(config["count"]))
    return tuple((cos(angle * pi / 180.0), 0.0, sin(angle * pi / 180.0)) for angle in angles)


def antipodal_fibonacci(config):
    assert config["count"] > 0
    assert config["count"] % 2 == 0
    pair_count = config["count"] // 2
    golden_angle = pi * (3.0 - sqrt(5.0))
    directions = []
    for index in range(pair_count):
        y = (index + 0.5) / pair_count
        radius = sqrt(1.0 - y * y)
        azimuth = index * golden_angle
        direction = (radius * cos(azimuth), y, radius * sin(azimuth))
        directions.extend((direction, tuple(-value for value in direction)))
    return directions


def latitude_rings(config):
    latitudes = config["latitudes_degrees"]
    counts = config["counts"]
    assert len(latitudes) > 0
    assert len(latitudes) == len(counts)
    assert all(-90.0 < latitude < 90.0 for latitude in latitudes)
    assert all(type(count) is int and count > 0 for count in counts)
    directions = []
    for latitude, count in zip(latitudes, counts, strict=True):
        elevation = latitude * pi / 180.0
        for index in range(count):
            azimuth = (config["azimuth_start_degrees"] + index * 360.0 / count) * pi / 180.0
            directions.append((cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth)))
    return directions


def clean(value):
    return 0.0 if abs(value) < 1.0e-15 else value


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
parser.add_argument("output", type=Path)
arguments = parser.parse_args()
assert arguments.config.resolve() != arguments.output.resolve()

with arguments.config.open("rb") as stream:
    values = tomllib.load(stream)
generator = values["jet_generator"]
assert generator["inner_radius"] >= 0.0

directions = {
    "planar": planar,
    "antipodal_fibonacci": antipodal_fibonacci,
    "latitude_rings": latitude_rings,
}[generator["type"]](generator)
radius = generator["inner_radius"]
tables = "\n\n".join(f"[[jets]]\nposition = {[clean(radius * value) for value in outward]}\ndirection = {[clean(-value) for value in outward]}" for outward in directions)
prefix = arguments.config.read_text().partition("[jet_generator]")[0].rstrip()
arguments.output.write_text(f"{prefix}\n\n{tables}\n")
