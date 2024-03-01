from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="JITBeijingGeometry",
      include_paths=["include"], ext_modules=[
        CUDAExtension(
            "JITBeijingGeometry",
            ["jit.cpp", "kernel/forwardKernel.cu", "kernel/backwardKernel.cu"]
        )
    ],
      zip_safe=False,
      cmdclass={
          "build_ext": BuildExtension
      })