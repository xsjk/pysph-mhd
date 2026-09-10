"""Load PySPH frames and keep interpolation keyframes on the GPU."""

import math

import cuda.bindings.runtime as cudart
import cupy as cp
import h5py
import numpy as np

ARGON_MOLAR_MASS = 0.039948
GAS_CONSTANT = 8.31
ARROW_COUNT = 144
FIELDS = ("gid", "x", "y", "z", "u", "v", "w", "h", "m", "rho", "p", "Bx", "By", "Bz")

PACK_FRAME_CUDA = r"""
extern "C" __global__
void pack_frame(const unsigned int* gid, const double* x, const double* y,
                const double* z, const double* h, const double* mass,
                const double* rho, const double* molar_mass,
                const double* pressure, const double* bx, const double* by,
                const double* bz, float* output, int count, float gas_constant) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    int j = ((int)gid[i]) * 10;
    float density = (float)rho[i];
    float temperature = (float)pressure[i] * (float)molar_mass[i]
                        / (density * gas_constant);
    output[j] = (float)x[i];
    output[j + 1] = (float)y[i];
    output[j + 2] = (float)z[i];
    output[j + 3] = (float)h[i];
    output[j + 4] = sqrtf((float)(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]));
    output[j + 5] = logf(temperature);
    output[j + 6] = (float)bx[i];
    output[j + 7] = (float)bz[i];
    output[j + 8] = (float)(mass[i] / rho[i]);
    output[j + 9] = density;
}
"""


def frame_paths(input_dir):
    assert input_dir.is_dir(), f"input directory does not exist: {input_dir}"
    paths = sorted(input_dir.glob("*.hdf5"))
    assert len(paths) >= 2
    return paths


def case_name(paths):
    prefix, separator, suffix = paths[0].stem.rpartition("_")
    assert separator
    assert suffix.isdigit()
    assert all(path.stem.rpartition("_")[0] == prefix for path in paths)
    return prefix


def animation_times(source_times, frame_count, hold_frames):
    moving = np.linspace(source_times[0], source_times[-1], frame_count - 2 * hold_frames)
    return np.concatenate((np.full(hold_frames, source_times[0]), moving, np.full(hold_frames, source_times[-1])))


def read_frame(path):
    with h5py.File(path) as handle:
        arrays = handle["particles/fluid/arrays"]
        assert all(field in arrays for field in FIELDS)
        data = {field: arrays[field][...] for field in FIELDS}
        count = len(data["gid"])
        data["molar_mass"] = arrays["molar_mass"][...] if "molar_mass" in arrays else np.full(count, ARGON_MOLAR_MASS)
        time = float(handle["solver_data"].attrs["t"])
    assert all(len(values) == count for values in data.values())
    gids = data["gid"]
    assert gids.dtype == np.uint32
    assert np.array_equal(np.sort(gids), np.arange(count, dtype=np.uint32))
    assert all(np.isfinite(data[field]).all() for field in FIELDS[1:])
    return time, data


class GpuFrames:
    def __init__(self, context, paths):
        assert cudart.cudaSetDevice(0)[0] == cudart.cudaError_t.cudaSuccess
        self.pack_frame = cp.RawKernel(PACK_FRAME_CUDA, "pack_frame")
        self.data = []
        self.arrow_data = []
        times, statistics, detail_radii = [], [], []
        magnetic_limits, temperature_limits, density_limits = [], [], []

        first_time, first = read_frame(paths[0])
        self.particle_count = len(first["gid"])
        distance = np.sqrt(first["x"] ** 2 + first["y"] ** 2 + first["z"] ** 2)
        with h5py.File(paths[0]) as handle:
            arrays = handle["particles/fluid/arrays"]
            self.reference_radius = float(np.max(distance[arrays["material_id"][...] == 0])) if "material_id" in arrays else float(np.min(distance))
        self.view_radius = 1.08 * float(np.max(distance))
        grid_width = math.isqrt(ARROW_COUNT)
        assert grid_width * grid_width == ARROW_COUNT

        for index, path in enumerate(paths):
            time, data = (first_time, first) if index == 0 else read_frame(path)
            assert len(data["gid"]) == self.particle_count
            rho = data["rho"]
            pressure = data["p"]
            temperature = pressure * data["molar_mass"] / (rho * GAS_CONSTANT)
            magnetic = np.sqrt(data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2)
            speed = np.sqrt(data["u"] ** 2 + data["v"] ** 2 + data["w"] ** 2)
            focus = distance <= self.view_radius if index == 0 else (data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2 <= self.view_radius**2)
            positive_magnetic = magnetic[focus & (magnetic > 0.0)]
            assert len(positive_magnetic) > 0
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
            arrows, detail_radius = self._arrows(data, focus, grid_width)
            self.arrow_data.append(context.buffer(arrows.tobytes()))
            detail_radii.append(detail_radius)
            times.append(time)
            print(f"\rloaded {index + 1}/{len(paths)}", end="", flush=True)
        print()

        self.times = np.asarray(times)
        assert np.all(np.diff(self.times) > 0)
        self.detail_radii = np.asarray(detail_radii)
        self.magnetic_range = self._range(magnetic_limits)
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
        sliced = focus & (np.abs(data["y"]) <= 2.0 * data["h"])
        radius = np.sqrt(data["x"][sliced] ** 2 + data["z"][sliced] ** 2)
        detail_radius = float(
            np.clip(
                1.08 * np.quantile(radius + 2.0 * data["h"][sliced], 0.995),
                0.2 * self.reference_radius,
                self.view_radius,
            )
        )
        cell_width = 2.0 * detail_radius / grid_width
        centers = np.linspace(-detail_radius + cell_width / 2, detail_radius - cell_width / 2, grid_width)
        arrow_x, arrow_z = np.meshgrid(centers, centers)
        columns = ((data["x"][sliced] + detail_radius) / cell_width).astype(np.int32)
        rows = ((data["z"][sliced] + detail_radius) / cell_width).astype(np.int32)
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
            for field in ("Bx", "Bz")
        ]
        arrows = np.zeros((ARROW_COUNT, 8), dtype="f4")
        arrows[:, 0], arrows[:, 2] = arrow_x.ravel(), arrow_z.ravel()
        arrows[:, 4] = np.hypot(*magnetic)
        arrows[:, 6], arrows[:, 7] = magnetic
        return arrows, detail_radius

    def _upload_frame(self, context, data):
        size = self.particle_count * 10 * 4
        buffer = context.buffer(reserve=size)
        status, resource = cudart.cudaGraphicsGLRegisterBuffer(buffer.glo, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
        assert status == cudart.cudaError_t.cudaSuccess
        assert cudart.cudaGraphicsMapResources(1, resource, 0)[0] == cudart.cudaError_t.cudaSuccess
        status, pointer, mapped_size = cudart.cudaGraphicsResourceGetMappedPointer(resource)
        assert status == cudart.cudaError_t.cudaSuccess
        assert mapped_size == size
        device = [cp.asarray(data[field]) for field in ("gid", "x", "y", "z", "h", "m", "rho", "molar_mass", "p", "Bx", "By", "Bz")]
        memory = cp.cuda.UnownedMemory(int(pointer), size, buffer)
        output = cp.ndarray((self.particle_count, 10), dtype=cp.float32, memptr=cp.cuda.MemoryPointer(memory, 0))
        self.pack_frame(
            ((self.particle_count + 255) // 256,),
            (256,),
            (*device, output, self.particle_count, np.float32(GAS_CONSTANT)),
        )
        cp.cuda.get_current_stream().synchronize()
        del output, memory
        assert cudart.cudaGraphicsUnmapResources(1, resource, 0)[0] == cudart.cudaError_t.cudaSuccess
        assert cudart.cudaGraphicsUnregisterResource(resource)[0] == cudart.cudaError_t.cudaSuccess
        return buffer
