"""Single-session OpenGL-to-NVENC zero-copy encoder."""

import ast
import importlib

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart

VIDEO_TIMEBASE = 90_000

# PyNvVideoCodec 2.2.0 still imports the removed, unused Python 3.14 alias.
ast.Str = ast.Constant
nvc = importlib.import_module("PyNvVideoCodec")
del ast.Str


class CudaPlane:
    def __init__(self, height, width):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (height, width, 4),
            "strides": (width * 4, 4, 1),
            "typestr": "|u1",
            "data": (0, False),
        }

    def point_to(self, pointer):
        self.__cuda_array_interface__["data"] = (pointer, False)


class CudaFrame:
    def __init__(self, height, width):
        self.plane = CudaPlane(height, width)

    def cuda(self):
        return self.plane


class CudaNvencEncoder:
    def __init__(self, context, width, height, fps):
        assert nvc.__version__ == "2.2.0"
        status, count, devices = cudart.cudaGLGetDevices(1, cudart.cudaGLDeviceList.cudaGLDeviceListAll)
        assert status == cudart.cudaError_t.cudaSuccess
        assert count == 1
        assert devices == [0]
        assert cudart.cudaSetDevice(0)[0] == cudart.cudaError_t.cudaSuccess
        status, cuda_context = cuda.cuCtxGetCurrent()
        assert status == cuda.CUresult.CUDA_SUCCESS
        assert int(cuda_context) != 0

        self.width, self.height, self.fps = width, height, fps
        self.target = context.buffer(reserve=width * height * 4)
        status, self.resource = cudart.cudaGraphicsGLRegisterBuffer(
            self.target.glo,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
        )
        assert status == cudart.cudaError_t.cudaSuccess
        self.frame = CudaFrame(height, width)
        use_cpu_input = False
        self.encoder = nvc.CreateEncoder(
            width,
            height,
            "ABGR",
            use_cpu_input,
            cudacontext=int(cuda_context),
            cudastream=0,
            gpu_id=0,
            codec="h264",
            fps=fps,
            preset="P6",
            tuning_info="high_quality",
            rc="constqp",
            constqp=18,
            bf=3,
            gop=240,
            colorspace="bt709",
        )
        self.sequence_parameters = self.encoder.GetSequenceParams()
        self.packets = []

    def encode(self, index):
        assert cudart.cudaGraphicsMapResources(1, self.resource, 0)[0] == cudart.cudaError_t.cudaSuccess
        status, pointer, size = cudart.cudaGraphicsResourceGetMappedPointer(self.resource)
        assert status == cudart.cudaError_t.cudaSuccess
        assert size == self.width * self.height * 4
        self.frame.plane.point_to(int(pointer))
        parameters = nvc.NV_ENC_PIC_PARAMS()
        parameters.inputTimeStamp = index
        for packet in self.encoder.Encode(self.frame, parameters):
            self.packets.append((bytes(packet["data"]), packet["picture_type"], packet["timestamp"]))
        assert cudart.cudaGraphicsUnmapResources(1, self.resource, 0)[0] == cudart.cudaError_t.cudaSuccess

    def finish(self, output):
        for packet in self.encoder.EndEncode():
            self.packets.append((bytes(packet["data"]), packet["picture_type"], packet["timestamp"]))
        muxer = nvc.FFmpegMuxer(
            file_path=str(output),
            media_format=nvc.GetMediaFormat(str(output)),
            codec="h264",
            width=self.width,
            height=self.height,
            fps_num=self.fps,
            fps_den=1,
            timebase_num=1,
            timebase_den=VIDEO_TIMEBASE,
            extradata=self.sequence_parameters,
        )
        muxer.SetUniformPtsIncrement(VIDEO_TIMEBASE // self.fps)
        for data, picture_type, timestamp in self.packets:
            muxer.MuxVideoPacket(data, picture_type, timestamp)
        muxer.Finalize()
        del muxer
        assert cudart.cudaGraphicsUnregisterResource(self.resource)[0] == cudart.cudaError_t.cudaSuccess
        self.target.release()
