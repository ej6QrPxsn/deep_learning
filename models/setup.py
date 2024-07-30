import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

setup(
    name='llama',
    ext_modules=[
        CUDAExtension('llama', [
            'llama.cpp',
            'flash_attention.cu',
        ],
            extra_compile_args=['-std=c++17']
        )
    ],
    version="2.0.1",
    cmdclass={
        'build_ext': BuildExtension
    })
