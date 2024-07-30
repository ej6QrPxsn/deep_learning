from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention',
    ext_modules=[
        CUDAExtension('flash_attention', [
            'flash_attention.cpp',
            'flash_attention_kernel.cu',
        ],
            extra_compile_args=['-std=c++17']
        )
    ],
    version="2.0.1",
    cmdclass={
        'build_ext': BuildExtension
    })
