from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="JITSelfCalibration",
      include_paths=["include"], ext_modules=[
        CUDAExtension(
            "JITSelfCalibration",
            ["jit.cpp", "kernel/differentiableFanFlatGradient.cu", 
             "kernel/differentiableConeGradient.cu", "kernel/backward.cu", "kernel/forward.cu"]
        )
    ], zip_safe=False, cmdclass={
        "build_ext": BuildExtension
    })