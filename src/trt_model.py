import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class HostDeviceMemory:
    def __init__(self, host, device):
        self.host = host
        self.device = device


class TensorRTModel:

    dynamic_input = False

    def __init__(self, logger: trt.Logger):
        self.TRT_LOGGER = logger
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def __call__(self, img: np.ndarray):
        out = self.run_inference(self.preprocess(img))
        return self.postprocess(out)

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMemory(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMemory(host_mem, device_mem))
        return inputs, outputs, bindings

    def run_inference(self, img):
        if not self.context.all_binding_shapes_specified or self.dynamic_input:
            self.context.set_binding_shape(0, img.shape)
        inputs, outputs, bindings = self.allocate_buffers()
        inputs[0].host = img
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async(bindings=bindings, stream_handle=self.stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        return [out.host for out in outputs]

    def get_engine(self) -> trt.ICudaEngine:
        raise NotImplementedError

    def preprocess(self, img: np.ndarray):
        raise NotImplementedError

    def postprocess(self, out: list):
        raise NotImplementedError
