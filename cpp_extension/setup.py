from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='splatter_cpp',
    ext_modules=[
        CppExtension('splatter_cpp', ['splatter.cpp'],
        extra_compile_args=['-O3']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })