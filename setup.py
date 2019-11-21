from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import glob as gg
import os
import sys

with open('README.md') as fh:
      readme = fh.read()

expt = ["expt.pyx", "Matrix.cpp", "Data.cpp", "MutualInformationMatrix.cpp", "Filter.cpp", "Math.cpp"]

extensions = Extension(
    "expt", 
    sources = ["pymrmre/expt/" + str for str in expt], 
    language = "c++",
    extra_link_args=['-fopenmp'],
    extra_compile_args=["-fopenmp", "-Ofast"],
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
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=[
        'cython>=0.25'
    ],
    license='MIT license',
    zip_safe=False,
    keywords='pymrmre featureselection genomics computationalbiology',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
    )
