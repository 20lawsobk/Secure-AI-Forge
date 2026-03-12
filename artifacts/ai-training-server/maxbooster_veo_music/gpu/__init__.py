from ai_model.gpu.hyper_core import HyperGPU, PrecisionMode

gpu = HyperGPU(lanes=512, tensor_cores=8, precision=PrecisionMode.MIXED)

__all__ = ["gpu", "HyperGPU", "PrecisionMode"]
