"""Load PySPH frames and keep interpolation keyframes on the GPU."""

import math

import cuda.bindings.runtime as cudart
import cupy as cp
import h5py
import numpy as np
from scipy.constants import Boltzmann, proton_mass

ARGON_MOLAR_MASS = 0.039948
GAS_CONSTANT = 8.31
ARROW_COUNT = 144
FIELDS = ("gid", "x", "y", "z", "u", "v", "w", "h", "m", "rho", "p", "Bx", "By", "Bz")

PACK_FRAME_CUDA = r"""
extern "C" __global__
void pack_frame(const unsigned int* gid, const double* x, const double* y,
                const double* z, const double* h, const double* mass,
                const double* rho, const double* temperature_factor,
                const double* pressure, const double* bx, const double* by,
                const double* bz, float* output, int count) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    int j = ((int)gid[i]) * 10;
    float density = (float)rho[i];
    float temperature = (float)pressure[i] * (float)temperature_factor[i] / density;
    output[j] = (float)x[i];
    output[j + 1] = (float)y[i];
    output[j + 2] = (float)z[i];
    output[j + 3] = (float)h[i];
    output[j + 4] = sqrtf((float)(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]));
    output[j + 5] = logf(temperature);
    output[j + 6] = (float)bx[i];
    output[j + 7] = (float)by[i];
    output[j + 8] = (float)(mass[i] / rho[i]);
    output[j + 9] = density;
}
"""


def frame_paths(input_dir):
    assert input_dir.is_dir(), f"input directory does not exist: {input_dir}"
    paths = sorted(input_dir.glob("*.hdf5"))
    assert len(paths) >= 2
    preflight = sorted((input_dir / "argon_preflight").glob("*.hdf5"))
    if preflight:
        with h5py.File(preflight[-1]) as before, h5py.File(paths[0]) as after:
            np.testing.assert_allclose(before["solver_data"].attrs["t"], after["solver_data"].attrs["t"], rtol=1.0e-6, atol=0.0)
        paths = preflight[:-1] + paths
    return paths


def case_name(paths):
    prefix, separator, suffix = paths[-1].stem.rpartition("_")
    assert separator
    assert suffix.isdigit()
    assert all(path.stem.rpartition("_")[0] == prefix for path in paths if path.parent == paths[-1].parent)
    return prefix


def animation_times(source_times, frame_count, hold_frames):
    moving = np.linspace(source_times[0], source_times[-1], frame_count - 2 * hold_frames)
    return np.concatenate((np.full(hold_frames, source_times[0]), moving, np.full(hold_frames, source_times[-1])))


def read_frame(path):
    with h5py.File(path) as handle:
        names = ("fluid", "hydrogen") if "hydrogen" in handle["particles"] else ("fluid",)
        parts = []
        offset = 0
        for name in names:
            arrays = handle[f"particles/{name}/arrays"]
            assert all(field in arrays for field in FIELDS)
            part = {field: arrays[field][...] for field in FIELDS}
            count = len(part["gid"])
            np.testing.assert_array_equal(np.sort(part["gid"]), np.arange(count))
            part["gid"] += offset
            offset += count
            part["molar_mass"] = arrays["molar_mass"][...] if "molar_mass" in arrays and arrays["molar_mass"].size > 0 else np.full(count, ARGON_MOLAR_MASS)
            part["temperature_factor"] = part["molar_mass"] / GAS_CONSTANT
            if "material_id" in arrays and arrays["material_id"].size > 0:
                part["temperature_factor"][arrays["material_id"][...] == 2] = proton_mass / (2.0 * Boltzmann)
            parts.append(part)
        data = {field: np.concatenate([part[field] for part in parts]) for field in parts[0]}
        count = len(data["gid"])
        time = float(handle["solver_data"].attrs["t"])
    assert all(len(values) == count for values in data.values())
    gids = data["gid"]
    assert gids.dtype == np.uint32
    assert np.array_equal(np.sort(gids), np.arange(count, dtype=np.uint32))
    assert all(np.isfinite(data[field]).all() for field in FIELDS[1:])
    return time, data


class GpuFrames:
    def __init__(self, context, paths, detail_radius):
        assert np.isfinite(detail_radius)
        assert detail_radius > 0.0
        self.detail_radius = detail_radius
        assert cudart.cudaSetDevice(0)[0] == cudart.cudaError_t.cudaSuccess
        self.pack_frame = cp.RawKernel(PACK_FRAME_CUDA, "pack_frame")
        self.data = []
        self.arrow_data = []
        times, statistics, particle_counts = [], [], []
        magnetic_limits, temperature_limits, density_limits = [], [], []

        first_time, first = read_frame(paths[0])
        distance = np.sqrt(first["x"] ** 2 + first["y"] ** 2 + first["z"] ** 2)
        self.view_radius = 1.08 * float(np.max(distance))
        grid_width = math.isqrt(ARROW_COUNT)
        assert grid_width * grid_width == ARROW_COUNT

        for index, path in enumerate(paths):
            time, data = (first_time, first) if index == 0 else read_frame(path)
            particle_counts.append(len(data["gid"]))
            rho = data["rho"]
            pressure = data["p"]
            temperature = pressure * data["temperature_factor"] / rho
            magnetic = np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
            speed = np.sqrt(data["u"] ** 2 + data["v"] ** 2 + data["w"] ** 2)
            focus = distance <= self.view_radius if index == 0 else (data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2 <= self.view_radius**2)
            positive_magnetic = magnetic[focus & (magnetic > 0.0)]
            if len(positive_magnetic) > 0:
                magnetic_limits.append(np.quantile(positive_magnetic, (0.001, 0.999)))
            temperature_limits.append(np.quantile(temperature[focus], (0.001, 0.999)))
            density_limits.append(np.quantile(rho[focus], (0.001, 0.999)))
            statistics.append(
                (
                    np.quantile(magnetic[focus], 0.99),
                    np.quantile(temperature[focus], 0.99),
                    np.quantile(pressure[focus], 0.99),
                    np.median(speed[focus]),
                )
            )
            self.data.append(self._upload_frame(context, data))
            arrows = self._arrows(data, focus, grid_width)
            self.arrow_data.append(context.buffer(arrows.tobytes()))
            times.append(time)
            print(f"\rloaded {index + 1}/{len(paths)}", end="", flush=True)
        print()

        self.times = np.asarray(times)
        assert np.all(np.diff(self.times) > 0)
        self.particle_counts = np.asarray(particle_counts)
        assert np.all(np.diff(self.particle_counts) >= 0), "Only appended particle IDs are supported at a launch event."
        self.magnetic_range = self._range(magnetic_limits) if magnetic_limits else None
        if self.magnetic_range is not None:
            lower, upper = self.magnetic_range
            # Keep numerical field tails from consuming the entire logarithmic scale.
            self.magnetic_range = max(lower, upper * 1.0e-4), upper
        self.temperature_range = self._range(temperature_limits)
        self.density_range = self._range(density_limits)
        self.statistics = np.asarray(statistics, dtype="f4")
        self.statistics /= np.maximum(np.max(self.statistics, axis=0), 1.0e-30)

    @staticmethod
    def _range(limits):
        values = np.asarray(limits)
        result = float(np.min(values[:, 0])), float(np.max(values[:, 1]))
        assert 0.0 < result[0] < result[1]
        return result

    def _arrows(self, data, focus, grid_width):
        sliced = focus & (np.abs(data["z"]) <= 2.0 * data["h"])
        half_width, half_height = self.detail_radius * 640 / 360, self.detail_radius
        cell_width, cell_height = 2.0 * half_width / grid_width, 2.0 * half_height / grid_width
        arrow_x, arrow_y = np.meshgrid(
            np.linspace(-half_width + cell_width / 2, half_width - cell_width / 2, grid_width),
            np.linspace(-half_height + cell_height / 2, half_height - cell_height / 2, grid_width),
        )
        columns = np.floor((data["x"][sliced] + half_width) / cell_width).astype(np.int32)
        rows = np.floor((data["y"][sliced] + half_height) / cell_height).astype(np.int32)
        inside = (columns >= 0) & (columns < grid_width) & (rows >= 0) & (rows < grid_width)
        cells = rows[inside] * grid_width + columns[inside]
        volume = (data["m"][sliced] / data["rho"][sliced])[inside]
        volume_sum = np.bincount(cells, weights=volume, minlength=ARROW_COUNT)
        magnetic = [
            np.divide(
                np.bincount(
                    cells,
                    weights=data[field][sliced][inside] * volume,
                    minlength=ARROW_COUNT,
                ),
                volume_sum,
                out=np.zeros(ARROW_COUNT),
                where=volume_sum > 0,
            )
            for field in ("Bx", "By")
        ]
        arrows = np.zeros((ARROW_COUNT, 8), dtype="f4")
        arrows[:, 0], arrows[:, 1] = arrow_x.ravel(), arrow_y.ravel()
        arrows[:, 4] = np.hypot(*magnetic)
        arrows[:, 6], arrows[:, 7] = magnetic
        return arrows

    def _upload_frame(self, context, data):
        count = len(data["gid"])
        size = count * 10 * 4
        buffer = context.buffer(reserve=size)
        status, resource = cudart.cudaGraphicsGLRegisterBuffer(buffer.glo, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
        assert status == cudart.cudaError_t.cudaSuccess
        assert cudart.cudaGraphicsMapResources(1, resource, 0)[0] == cudart.cudaError_t.cudaSuccess
        status, pointer, mapped_size = cudart.cudaGraphicsResourceGetMappedPointer(resource)
        assert status == cudart.cudaError_t.cudaSuccess
        assert mapped_size == size
        device = [cp.asarray(data[field]) for field in ("gid", "x", "y", "z", "h", "m", "rho", "temperature_factor", "p", "Bx", "By", "Bz")]
        memory = cp.cuda.UnownedMemory(int(pointer), size, buffer)
        output = cp.ndarray((count, 10), dtype=cp.float32, memptr=cp.cuda.MemoryPointer(memory, 0))
        self.pack_frame(
            ((count + 255) // 256,),
            (256,),
            (*device, output, count),
        )
        cp.cuda.get_current_stream().synchronize()
        del output, memory
        assert cudart.cudaGraphicsUnmapResources(1, resource, 0)[0] == cudart.cudaError_t.cudaSuccess
        assert cudart.cudaGraphicsUnregisterResource(resource)[0] == cudart.cudaError_t.cudaSuccess
        return buffer
