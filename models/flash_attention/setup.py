from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='flash_attention_cpp',
      ext_modules=[cpp_extension.CppExtension('flash_attention_cpp', ['flash_attention.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
