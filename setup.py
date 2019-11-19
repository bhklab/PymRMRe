from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import glob as gg
import os
import sys

with open('README.md') as fh:
      readme = fh.read()

sources = ["expt.pyx", "Matrix.cpp", "Data.cpp", "MutualInformationMatrix.cpp", "Filter.cpp", "Math.cpp"]

extensions = Extension(
    "expt", 
    sources = ["expt/" + str for str in sources], 
    language = "c++", 
    extra_compile_args = ["-stdlib=libc++"],
    extra_link_args = ["-stdlib=libc++"],
    )

requirements = [
    'numpy',
    'pandas'
]

if sys.platform == 'darwin':
  os.environ['CC'] = 'clang-omp'
  os.environ['CXX'] = 'clang-omp++'

setup(
    name = "pymrmre",
    version="0.1.0",
    description="A Python package for Parallelized Minimum Redundancy, Maximum Relevance (mRMR) Ensemble Feature selections.",
    longdescription=readme,
    url="https://github.com/bhklab/PymRMRe",
    author="Bo Li, Benjamin Haibe-Kains",
    author_email="benjamin.haibe.kains@utoronto.ca",
    packages=[
        'expt',
    ],
    include_package_data=True,
    install_requires=requirements,
    setup_requires=[
        'cython>=0.25'
    ],
    license='MIT license',
    zip_safe=False,
    keywords='pymrmre featureselection genomics computationalbiology',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Langauge :: Python :: 3.6'
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
    )
