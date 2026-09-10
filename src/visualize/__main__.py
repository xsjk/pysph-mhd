"""Render PySPH output as a 4K, 60 FPS GPU-encoded video."""

import argparse
from pathlib import Path
from time import perf_counter

import moderngl

from .data import GpuFrames, animation_times, case_name, frame_paths
from .encoder import CudaNvencEncoder
from .renderer import FPS, FRAME_COUNT, HEIGHT, WIDTH, Renderer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="input_dir")
    parser.add_argument("--output", type=Path, required=True, dest="output_dir")
    return parser.parse_args()


def main():
    arguments = parse_args()
    paths = frame_paths(arguments.input_dir)
    name = case_name(paths)
    context = moderngl.create_standalone_context(require=430, backend="egl")
    assert "NVIDIA" in context.info["GL_RENDERER"].upper()
    frames = GpuFrames(context, paths)
    renderer = Renderer(context, frames, name)
    encoder = CudaNvencEncoder(context, WIDTH, HEIGHT, FPS)
    times = animation_times(frames.times, FRAME_COUNT, FPS // 2)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output = arguments.output_dir / f"{name}.mp4"

    start = perf_counter()
    for index, time in enumerate(times):
        renderer.render_into(float(time), encoder.target)
        context.finish()
        encoder.encode(index)
        if (index + 1) % FPS == 0:
            print(f"\rencoded {index + 1}/{FRAME_COUNT}", end="", flush=True)
    encoder.finish(output)
    elapsed = perf_counter() - start
    print(f"\nwrote {output} in {elapsed:.2f} s ({FRAME_COUNT / elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
