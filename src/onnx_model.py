import onnxruntime as ort
import numpy as np

from typing import List


class OnnxModel:

    session: ort.InferenceSession
    output_shape: List[int]
    input_shape: List[int]
    input_name: str
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        ('CPUExecutionProvider', {
            'use_arena': True,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cpu_fast_math_mode': True,
        }),
    ]

    def __init__(self, path: str):
        if not path:
            raise ValueError('path must be a valid path')
        self.session = ort.InferenceSession(path, providers=self.providers)
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        print("%s session started" % path)

    def __call__(self, img: np.ndarray):
        inp = self.preprocess(img)
        out = self.feed(inp)
        return self.postprocess(out)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def feed(self, inp: np.ndarray):
        out = self.session.run(None, {self.input_name: inp})
        return out

    def postprocess(self, out: List):
        raise NotImplementedError
