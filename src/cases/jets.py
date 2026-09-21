"""Element-independent gun layouts, orientations, and particle labels."""

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import override

import numpy as np

from ..application import MHDApplication


@dataclass(frozen=True)
class LatitudeRingsConfig:
    radius: float
    latitudes_degrees: list[float]
    counts: list[int]
    azimuth_start_degrees: float

    def validate(self):
        assert isfinite(self.radius)
        assert self.radius > 0.0
        assert isfinite(self.azimuth_start_degrees)
        assert len(self.latitudes_degrees) == len(self.counts) > 0
        assert all(isfinite(latitude) and -90.0 <= latitude <= 90.0 for latitude in self.latitudes_degrees)
        assert all(type(count) is int and count > 0 for count in self.counts)
        assert all(abs(latitude) < 90.0 or count == 1 for latitude, count in zip(self.latitudes_degrees, self.counts, strict=True))


def latitude_rings(layout):
    """Return gun fronts and inward axes, ordered clockwise viewed from +z.

    Returns
    -------
    tuple
        Arrays of front positions and unit inward directions, one row per gun.
    """
    layout.validate()
    outward = []
    for latitude, count in zip(layout.latitudes_degrees, layout.counts, strict=True):
        elevation = latitude * pi / 180.0
        for index in range(count):
            azimuth = (layout.azimuth_start_degrees - index * 360.0 / count) * pi / 180.0
            outward.append((cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation)))
    outward = np.asarray(outward)
    outward[np.abs(outward) < 1.0e-15] = 0.0
    return layout.radius * outward, -outward


def jet_basis(direction):
    """Map local transverse x/y and inward z to a right-handed global frame.

    Returns
    -------
    ndarray
        Proper rotation matrix with local basis vectors as columns.
    """
    axis = np.asarray(direction)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(axis[2]) > 0.9:
        reference = np.asarray((0.0, np.sign(axis[2]), 0.0))
    transverse_y = reference - np.dot(reference, axis) * axis
    transverse_y /= np.linalg.norm(transverse_y)
    transverse_x = np.cross(transverse_y, axis)
    return np.column_stack((transverse_x, transverse_y, axis))


class JetApplication(MHDApplication):
    @override
    def create_particles(self):
        (particle,) = self.create_mhd_particles()
        assert particle.name == "fluid"
        source_id = particle.source_id.copy()
        self.scheme.setup_properties(particle)
        # The scheme removes extra properties; restore gun IDs before GPU upload.
        particle.add_property("source_id", type="int", data=source_id)
        particle.set_output_arrays([*particle.output_property_arrays, "source_id", "material_id", "molar_mass", "divBsymm", "divBdiff", "psi", "converged"])
        return [particle]
