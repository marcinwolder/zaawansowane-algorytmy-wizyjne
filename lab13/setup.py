# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='graphgen',
    ext_modules=[
        CppExtension(
            name='graphgen',
            sources=['graphgen.cpp'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++17',
                    # '-march=native',  # enable if you build for a fixed machine
                    # You can add '-fopenmp' here and link if you want OpenMP later
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
